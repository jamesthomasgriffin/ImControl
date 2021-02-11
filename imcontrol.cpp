#include "imcontrol.h"

#include "imcontrol_internal.h"


namespace ImControl {

    namespace Transformations {
        /*
        Implementation of the transformation stack

        There are three separate stacks of matrices, one for the matrix transformation,
        one for the first derivatives and one for the second derivatives.

        The variables which can be differentiated against can vary but only when the stacks are empty.

        By default pushing a transformation will calculate a new matrix in the stack, however the
        transformation is applied in place if using the composite transformation system.  This means
        that transformations cannot be popped during a composite transformation.
        */

        const ImMat4& TransformationStack::getDerivativeMatrix(param_t param) const
        {
            auto p = m_derivative_parameters.find(param);

            // By convention if a parameter isn't found we just return 0
            if (p == m_derivative_parameters.end())
                return m_zero_matrix;

            return beginDerivatives()[p - m_derivative_parameters.begin()];
        }

        const ImMat4& TransformationStack::getSecondDerivativeMatrix(param_t p1, param_t p2) const
        {
            // NB, we might want our own find here as we only need to look as far as m_num_second_derivs
            // However this is only relevant if computing a large number of first derivatives
            size_t ix1 = m_derivative_parameters.find(p1) - m_derivative_parameters.begin();

            if (ix1 >= m_num_second_derivs)
                return m_zero_matrix;

            if (p1 == p2)  // Shortcut in this common case
                return beginSecondDerivatives()[symmetric_index_ordered(ix1, ix1)];

            size_t ix2 = m_derivative_parameters.find(p2) - m_derivative_parameters.begin();

            if (ix2 >= m_num_second_derivs)
                return m_zero_matrix;

            return beginSecondDerivatives()[symmetric_index(ix1, ix2)];
        }

        // This must only be called when the stack is empty
        void TransformationStack::setDerivativeParameters(const ImVector<param_t>& d_params, int n_second_derivatives)
        {
            IM_ASSERT(isEmpty());
            IM_ASSERT(n_second_derivatives <= d_params.size());

            m_derivative_parameters = d_params;
            m_num_second_derivs = n_second_derivatives;

            // This is the only function where the above two function parameters should be changed and so the only place where m_width is recalculated
            m_width = 1 + (size_t)m_derivative_parameters.size() + symmetric_size(m_num_second_derivs);
        }

        void TransformationStack::copyWithinStack(size_t source_ix, size_t target_ix)
        {
            IM_ASSERT((source_ix < m_depth) && (target_ix < m_depth));

            if (source_ix == target_ix)
                return;

            memcpy(m_data.Data + target_ix * m_width, m_data.Data + source_ix * m_width, m_width * sizeof(ImMat4));
        }

        void TransformationStack::pushIdentity()
        {
            resize(m_depth + 1);  // NB increments m_depth

            if (m_depth > 1) {
                copyWithinStack(m_depth - 2, m_depth - 1);
                return;
            }

            // Deal with case when stack was empty
            m_data[0] = m_identity;

            // Set all derivative matrices to 0
            if (m_width > 1)
                memset(m_data.Data + 1, 0, ((size_t)m_width - 1) * sizeof(ImMat4));
        }

        template<typename T_t>
        void TransformationStack::pushConstantTransformation(const T_t& T)
        {
            if (m_composite_level == 0)
                pushIdentity();

            T.applyTransformationOnRightInPlace(beginLevel(), endLevel());
        }

        template<typename T_t>
        void TransformationStack::pushTransformation(const T_t& T, param_t param) 
        {
            pushTransformation(T, Parameters::Parameter{ param });
        }

        template<typename T_t>
        void TransformationStack::pushTransformation(const T_t& T, const Parameters::Parameter& param)
        {
            auto p = m_derivative_parameters.find(param.p);

            // If parameter is not being tracked then transformation may as well be constant
            if (p == m_derivative_parameters.end())
                return pushConstantTransformation(T);

            size_t param_ix = p - m_derivative_parameters.begin();

            if (m_composite_level == 0)
                pushIdentity();

            // We apply the transformation in place (we only have the previous
            // values available if m_composite_level is zero). Since the calculation
            // of the derivatives requires the previous transformation matrix and
            // the calculation of the second derivatives requires both the previous
            // transformation matrix and the previous derivative matrices, we will
            // first calculate the second derivatives, then the first derivatives
            // and only then the new transformation matrix.

            const ImMat4& old_M = getMatrix();  // A const reference so we cannot apply anything in-place by accident

            ImMat4 const* const old_D1 = beginDerivatives();  // Again this is for looking up derivative values without changing them

            // First just apply transformation matrix to all second derivatives
            T.applyTransformationOnRightInPlace(beginSecondDerivatives(), endSecondDerivatives());

            if (param_ix < m_num_second_derivs)
            {
                // D2 will increment through the second derivatives involving param_ix, starting with (0, param_ix)
                ImMat4* D2 = beginSecondDerivatives() + symmetric_index_ordered<size_t>(0, param_ix);

                for (size_t ix = 0; ix < param_ix; ++ix) {
                    *D2 += param.d * T.applyDerivativeOnRightInPlace(ImMat4{ old_D1[ix] });  // D2 = beginSecondDerivatives() + ix + (param_ix * (param_ix + 1)) / 2
                    ++D2;
                }

                // ix = param_ix case, D2 = beginSecondDerivatives() + param_ix + (param_ix * (param_ix + 1)) / 2
                *D2 += (2.0f * param.d) * T.applyDerivativeOnRightInPlace(ImMat4{ old_D1[param_ix] });
                *D2 += (param.d * param.d) * T.apply2ndDerivOnRightInPlace(ImMat4{ old_M });
                *D2 += param.d2 * T.applyDerivativeOnRightInPlace(ImMat4{ old_M });

                // ix > param_ix case
                for (size_t ix = param_ix + 1; ix < m_num_second_derivs; ++ix) {
                    D2 += ix;  // Larger increment needed now ix > param_ix
                    *D2 += param.d * T.applyDerivativeOnRightInPlace(ImMat4{ old_D1[ix] });  // D2 = beginSecondDerivatives() + param_ix + (ix * (ix + 1)) / 2
                }
                IM_ASSERT(D2 == beginSecondDerivatives() + symmetric_index_ordered(param_ix, m_num_second_derivs - 1));
            }

            // Apply to first derivatives
            T.applyTransformationOnRightInPlace(beginDerivatives(), endDerivatives());
            beginDerivatives()[param_ix] += param.d * T.applyDerivativeOnRightInPlace(ImMat4{ old_M });

            // Apply to transformation matrix
            T.applyTransformationOnRightInPlace(beginLevel(), beginLevel() + 1);
        }

        void TransformationStack::reset()
        {
            clear();

            m_derivative_parameters.clear();
            m_num_second_derivs = 0;
        }

        ImJet1<ImVec4> apply_stack_to_1jet(const TransformationStack& transform, const ImJet1<ImVec4>& jet)
        {
            ImJet1<ImVec4> output{ jet.parameters };

            output.value() = transform.apply(jet.value());

            for (int i = 0; i < jet.size(); ++i)
                output.derivative(i) = transform.applyDerivative(jet.value(), jet.parameters[i]);

            return output;
        }

        ImJet2<ImVec4> apply_stack_to_2jet(const TransformationStack& transform, const ImJet2<ImVec4>& jet)
        {
            ImJet2<ImVec4> output{ jet.parameters };

            output.value() = transform.apply(jet.value());

            for (int i = 0; i < jet.size(); ++i) {
                output.derivative(i) = transform.applyDerivative(jet.value(), jet.parameters[i]);
                for (int j = 0; j <= i; ++j)
                    output.second_derivative(j, i) = transform.applySecondDerivative(jet.value(), jet.parameters[i], jet.parameters[j]);
            }
            return output;
        }

        ImJet1<ImVec4> project_1jet(const ImJet1<ImVec4>& jet)
        {
            const ImVec4& p = jet.value();   // Call p the position
            const float& w = p.w;  // Call w the w-coord of p

            // Calculate the output 1-jet
            ImJet1<ImVec4> output{ jet.parameters };
            output.value() = p / w;

            // Now fill in the second derivatives
            for (int i = 0; i < jet.size(); ++i)
            {
                const ImVec4& dpdu = jet.derivative(i);  // u is a name for the ith parameter
                const float& dwdu = dpdu.w;

                // Quotient rule
                output.derivative(i) = (dpdu * w - p * dwdu) / (w * w);
            }
            return output;
        }

        ImJet2<ImVec4> project_2jet(const ImJet2<ImVec4>& jet)
        {
            const ImVec4& p = jet.value();   // Call p the position
            const float& w = p.w;  // Call w the w-coord of p

            // Calculate the output 1-jet
            ImJet2<ImVec4> output{ jet.parameters };
            output.value() = p / w;

            // Now fill in the second derivatives
            for (int i = 0; i < jet.size(); ++i)
            {
                const ImVec4& dpdu = jet.derivative(i);  // u is a name for the ith parameter
                const float& dwdu = dpdu.w;

                // Quotient rule
                output.derivative(i) = (dpdu * w - p * dwdu) / (w * w);

                for (int j = 0; j <= i; ++j)
                {
                    const ImVec4& dpdv = jet.derivative(j);  // v is a name for the jth parameter
                    const float& dwdv = dpdv.w;

                    const ImVec4& d2pdudv = jet.second_derivative(j, i);
                    const float& d2wdudv = d2pdudv.w;

                    // This is a formula for the second derivative of p / w with respect to u and v
                    output.second_derivative(j, i) = (d2pdudv * w * w - dpdu * dwdv * w - dpdv * dwdu * w - p * d2wdudv * w + p * (2 * dwdu * dwdv)) / (w * w * w);
                }
            }
            return output;
        }

        ImJet2<float> distance_squared(const ImJet2<ImVec4>& jet, const ImVec4& target)
        {
            ImJet2<float> output{ jet.parameters };
            ImVec4 diff = jet.value() - target;
            output.value() = dot(diff, diff) / 2;

            for (int i = 0; i < jet.size(); ++i) {
                output.derivative(i) = dot(diff, jet.derivative(i));
                for (int j = 0; j < jet.size(); ++j)
                    output.second_derivative(i, j) = dot(jet.derivative(i), jet.derivative(j)) + dot(diff, jet.second_derivative(i, j));
            }
            return output;
        }

        static inline void apply_second_derivative(const ImJet2<float>& jet, ImVector<float>& out, ImVector<float> const& x, float kappa = 0) {
            IM_ASSERT((jet.size() == out.size()) && (jet.size() == x.size()));
            // out <== out + f second_derivative * x + kappa * x
            for (int i = 0; i < jet.size(); ++i) {
                for (int j = 0; j < jet.size(); ++j) {
                    out[i] += jet.second_derivative(i, j) * x[j];
                }
                out[i] += kappa * x[i];
            }
        }

        ImVector<float> apply_conjugate_gradient_method(const ImJet2<float>& jet, float kappa) 
        {
            // Compute first_derivative / (second_derivative + kappa * I) using the conjugate gradient method.

            constexpr float error_tolerance{ 0.01f };
            const int D = jet.size();

            // Since we start with x = 0, the initialisation may look slightly different to standard
            ImVector<float> x{};
            x.resize(D, 0);  // Start with x = 0

            ImVector<float> r{};
            r.resize(D);  // Start with r = first_derivative
            for (int i = 0; i < D; ++i)
                r[i] = jet.derivative(i);

            ImVector<float> d{ r };

            float delta_new = 0;
            for (auto const& v : r)
                delta_new += v * v;

            float const delta_0 = delta_new > 0.001f ? delta_new : 0.001f;

            ImVector<float> q{};
            q.resize(D);

            for (int i = 0; (i < D) && (delta_new > error_tolerance * error_tolerance * delta_0); ++i)
            {
                for (auto& v : q)
                    v = 0.0f;  // Zero q
                apply_second_derivative(jet, q, d, kappa);

                float alpha{};
                for (int j = 0; j < D; ++j)
                    alpha += d[j] * q[j];
                alpha = delta_new / alpha;

                for (int j = 0; j < D; ++j) {
                    x[j] += alpha * d[j];
                    r[j] -= alpha * q[j];
                }

                float delta_old{ delta_new };

                delta_new = {};
                for (auto const& v : r)
                    delta_new += v * v;

                float beta = delta_new / delta_old;

                for (int j = 0; j < D; ++j)
                    d[j] = r[j] + beta * d[j];
            }
            return x;
        }

        inline ImVec4 calculate_tangent_after_projection(const TransformationStack& transform, const ImVec4& pos, param_t param)
        {
            ImVector<float*> params{};
            params.resize(1, param);
            ImJet1<ImVec4> jet{ params };
            jet.value() = transform.apply(pos);
            jet.derivative(0) = transform.applyDerivative(pos, param);

            ImJet1<ImVec4> output_jet = project_1jet(jet);
            return output_jet.derivative(0);
        }

        ImVector<float> bring_together(const TransformationStack& transform, const ImVec4& pos, const ImVec4& target, const ImVector<param_t>& free_parameters, float kappa)
        {
            // Minimises Q(a) = |p - Mv|^2 over a where Mv is the transformed vector, and
            // M(a) is the parametrised transformation with a the free parameter.
            // When p is closest to Mv, the derivative with respect to a will be
            // zero, so we use the Newton-Raphson method on the derivative.

            // A jet with given parameters, position and 0 first and second derivatives
            ImJet2<ImVec4> jet{ free_parameters };
            jet.value() = pos;

            // We need to project from 4 coords to 3 coords via p --> p / p.w
            ImJet2<ImVec4> output_jet = project_2jet(apply_stack_to_2jet(transform, jet));

            // Compute the distance squared
            ImJet2<float> Qderivatives = distance_squared(output_jet, target);

            // Now solve
            ImVector<float> steps = apply_conjugate_gradient_method(Qderivatives, kappa);

            return steps;
        }

    }  // namespace Transformations

    void ImControlContext::newFrame(Transformations::TransformationStack& transform)
    {
        IM_ASSERT(transform.getStackSize() == 0);

        ImVector<param_t> params{};

        // This is inefficient but the number of parameters is small
        for (auto p : m_registered_2nd_der_params)
            if (params.find(p) == params.end())
                params.push_back(p);

        int n_second_derivatives = params.size();

        for (auto p : m_registered_der_params)
            if (params.find(p) == params.end())
                params.push_back(p);

        // Pass the parameters for the duration of the following frame to the transform stack, second derivatives are computed for an initial segment of the vector
        transform.setDerivativeParameters(params, n_second_derivatives);

        m_registered_der_params.clear();
        m_registered_2nd_der_params.clear();
    }

    void ImControlContext::endFrame() 
    {
        // If there is a change in the deferral stack then apply it
        bool parameters_updated = m_deferral_stack.applyParameterChanges();

        // Reset deferral stack state
        m_deferral_stack.reset();

        if (m_activated_point_id) {
            ImGui::SetActiveID(m_activated_point_id, (ImGuiWindow*)m_activated_point_window);
            for(auto param : m_activated_point_parameters)
                registerFreeParameterSecondDerForNextFrame(param);
        }

        if (parameters_updated)
            ImGui::MarkItemEdited(ImGui::GetActiveID());

        // Reset the saved control point, NB z-position should not be reset
        m_activated_point_id = 0;
    }

    bool ImControlContext::beginView(const char* name, ImVec2 size, ImVec4 const& border_col, bool defer_changes)
    {
        IM_ASSERT(m_view_active == false);  // We cannot start two views at once

        ImGuiWindow* window = ImGui::GetCurrentWindow();
        if (window->SkipItems)
            return false;

        ImGuiContext& g = *GImGui;
        ImGui::PushID(name);
        m_view_id = window->GetID(0);

        // Handle automatic sizing - taking off space for border
        if (size.x <= 0)
            size.x = ImGui::GetContentRegionAvail().x - ((border_col.w > 0) ? 2.0f : 0.0f);
        if (size.y <= 0)
            size.y = ImGui::GetContentRegionAvail().y - ((border_col.w > 0) ? 2.0f : 0.0f);

        m_view_marker_size_factor = sqrtf(size.x * size.y);

        // Bounding box used for clipping 3D view and optional border
        m_cursor_pos_before_view = ImGui::GetCursorScreenPos();
        m_view_bb = ImRect{ m_cursor_pos_before_view, m_cursor_pos_before_view + size };

        if (border_col.w > 0.0f)
            m_view_bb.Max += ImVec2(2, 2);

        ImGui::ItemSize(m_view_bb);
        if (!ImGui::ItemAdd(m_view_bb, m_view_id))
        {
            ImGui::PopID();
            return false;
        }

        m_view_active = true;
        m_view_size = size;
        m_view_defers_changes = defer_changes;

        if (border_col.w > 0.0f)
        {
            window->DrawList->AddRect(m_view_bb.Min, m_view_bb.Max, ImGui::GetColorU32(border_col), 0.0f);
        }
        ImGui::PushClipRect(m_view_bb.Min, m_view_bb.Max, true);

        if (!m_view_defers_changes)
            pushDeferralSlot();

        return true;
    }

    bool ImControlContext::endView(bool include_button, ImGuiButtonFlags flags)
    {
        if (!m_view_active)
            return false;

        // Set the Dear ImGui state to where it should be after the view widget
        ImGui::PopID();
        ImGui::PopClipRect();

        if (!m_view_defers_changes)
            popDeferralSlot();

        bool ret_value = true;
        if (include_button) {
            bool hovered, held;
            ret_value = ImGui::ButtonBehavior(m_view_bb, m_view_id, &hovered, &held, flags);
        }

        ImGui::SetCursorScreenPos(m_cursor_position_after_view);

        m_view_id = 0;
        m_view_active = false;

        return ret_value;
    }

    void ImControlContext::pushDeferralSlot()
    {
        m_deferral_stack.pushDeferralSlot();
    }

    bool ImControlContext::popDeferralSlot()
    {
        return m_deferral_stack.popDeferralSlot();
    }

    ImVec2 ImControlContext::getViewBoundsMin() const
    {
        return m_cursor_pos_before_view;
    }

    ImVec2 ImControlContext::getViewBoundsMax() const
    {
        return m_cursor_pos_before_view + m_view_size;
    }

    const ImVec2& ImControlContext::getViewSize() const
    {
        return m_view_size;
    }


    void ImControlContext::drawDerivative(ImVec2 const& pos, ImVec2 const& d, ImVec4 const& col, float thickness) const
    {
        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        ImVec2 start_vector = viewCoordsToScreenCoords(pos);
        ImVec2 end_vector = viewCoordsToScreenCoords(pos + d);
        draw_list->AddLine(start_vector, end_vector, ImGui::GetColorU32(col), thickness);
    }

    ImVec2 ImControlContext::getMousePositionInViewCoords() const
    {
        ImVec2 mouse_pos = ImGui::GetMousePos();
        return screenCoordsToViewCoords(mouse_pos);
    }

    ImVec2 ImControlContext::viewCoordsToScreenCoords(ImVec2 const& v) const
    {
        ImVec2 result{ v };
        result += ImVec2(1, 1);
        result *= m_view_size / 2;
        result += m_cursor_pos_before_view;
        return result;
    }

    ImVec2 ImControlContext::screenCoordsToViewCoords(ImVec2 const& p) const
    {
        ImVec2 result{ p };
        result -= m_cursor_pos_before_view;
        result /= m_view_size / 2;
        result -= ImVec2(1, 1);
        return result;
    }

    bool ImControlContext::createPoint(const char* str_id, ImVec4 const& transformed_pos, ImControlPointFlags const& flags, ImGuiButtonFlags const& button_flags, float marker_radius, ImVec4 marker_col)
    {
        float w = transformed_pos.w;
        float z_order = transformed_pos.z / w;
        ImVec2 marker_pos = viewCoordsToScreenCoords({ transformed_pos.x / w, transformed_pos.y / w });

        // Calculate marker size in pixels
        if (!(flags & ImControlPointFlags_FixedSize))
            marker_radius /= w;
        if (!(flags & ImControlPointFlags_SizeInPixels))
            marker_radius *= m_view_marker_size_factor;

        // An Invisible button cannot have a size of 0, so limit radius to 1.0f
        marker_radius = (marker_radius < 1.0f) ? 1.0f : marker_radius;

        ImRect marker_bb(marker_pos - ImVec2{marker_radius, marker_radius}, marker_pos + ImVec2{ marker_radius, marker_radius });

        m_last_control_point_position = marker_pos;
        m_last_control_point_radius = marker_radius;
        m_last_step = {};

        if (z_order < -1.0f) // || (z_order > 1.0f), try this change to limit control points to viewing frustrum
        {
            // The control point is behind us so create a dummy icon we cannot interact with
            ImGui::Dummy({ 0, 0 });
            return false;
        }

        m_last_point_z_order = z_order;

        bool cp_activated = false;
        bool result{};

        {   // Create invisible button at marker and return cursor to its original position        
            ImVec2 saved_cursor_screen_pos = ImGui::GetCursorScreenPos();
            ImGui::SetCursorScreenPos(marker_bb.Min);

            // Use an invisible button to handle hovering / clicking / dragging
            result = ImGui::InvisibleButton(str_id, ImVec2{ 2 * marker_radius, 2 * marker_radius }, button_flags);

            ImGui::SetItemAllowOverlap();

            if (ImGui::IsItemActivated())  // Occurs only when first activated
            {
                cp_activated = true;
                ImVec2 vect_to_center = ImGui::GetMousePos() - (marker_bb.Min + marker_bb.Max) * 0.5f;
                vect_to_center /= marker_radius;  // Convert to coords local to marker
                float r2 = vect_to_center.x * vect_to_center.x + vect_to_center.y * vect_to_center.y;

                if (flags & ImControlPointFlags_ChooseClosestWhenOverlapping)
                    m_last_point_z_order = r2;

                if ((flags & ImControlPointFlags_Circular) && (r2 > 1.0f)) {
                    // Mouse is outside circle inscribed in bbox
                    cp_activated = false;
                    ImGui::ClearActiveID();
                }
            }

            ImGui::SetCursorScreenPos(saved_cursor_screen_pos);
        }

        // Draw the marker
        if ((flags & ImControlPointFlags_DrawControlPointMarkers) && marker_col.w > 0.0f)
        {
            if (flags & ImControlPointFlags_Circular)
            {
                if (ImGui::IsItemActive())
                    ImGui::GetWindowDrawList()->AddCircleFilled((marker_bb.Min + marker_bb.Max) / 2, marker_radius, ImGui::GetColorU32(marker_col));
                else
                    ImGui::GetWindowDrawList()->AddCircle((marker_bb.Min + marker_bb.Max) / 2, marker_radius, ImGui::GetColorU32(marker_col));
            }
            else
            {
                if (ImGui::IsItemActive())
                    ImGui::GetWindowDrawList()->AddRectFilled(marker_bb.Min, marker_bb.Max, ImGui::GetColorU32(marker_col));
                else
                    ImGui::GetWindowDrawList()->AddRect(marker_bb.Min, marker_bb.Max, ImGui::GetColorU32(marker_col), 0.0f);
            }
        }
        return result;
    }

    bool ImControlContext::saveActivatedPoint(const ImVector<float*>& params)
    {
        // If there is already a control point infront of this one then return
        if (m_activated_point_id && (m_last_point_z_order > m_activated_point_z_order))
            return false;

        m_activated_point_id = ImGui::GetItemID();
        m_activated_point_window = ImGui::GetCurrentWindow();
        m_activated_point_z_order = m_last_point_z_order;
        m_activated_point_parameters = params;
        return true;
    }

    bool ImControlContext::updateParameters(ImVector<parameter_change_t> const& changes, ImControlPointFlags const& flags)
    {
        m_last_step = changes;

        if (flags & ImControlPointFlags_DoNotChangeParams)
            return false;

        m_deferral_stack.addParameterChange(changes);

        if (flags & ImControlPointFlags_ApplyParamChangesImmediately)
        {
            bool change_applied = m_deferral_stack.applyParameterChanges();
            if (change_applied)
            {
                IM_ASSERT(ImGui::IsItemActive());  // If we are making parameter changes immediately the previous item must be active and marking it as edited makes sense
                ImGui::MarkItemEdited(ImGui::GetItemID());
            }
            return change_applied;
        }
        return false;
    }

    void ImControlContext::registerFreeParameterDerivativeForNextFrame(param_t param)
    {
        m_registered_der_params.push_back(param);
    }

    void ImControlContext::registerFreeParameterSecondDerForNextFrame(param_t p)
    {
        m_registered_2nd_der_params.push_back(p);
    }

    ImVec2 ImControlContext::pointInScreenCoords(const Transformations::TransformationStack& transform, const ImVec4& pos)
    {
        ImVec4 transformed_pos = transform.apply(pos);
        ImVec2 view_coords{ transformed_pos.x / transformed_pos.w, transformed_pos.y / transformed_pos.w };
        return viewCoordsToScreenCoords(view_coords);
    }

    bool ImControlContext::staticControlPoint(const char* str_id, const Transformations::TransformationStack& transform, const ImVec4& pos, ImControlPointFlags flags, ImGuiButtonFlags button_flags, float marker_radius, ImVec4 marker_col)
    {
        ImVec4 transformed_pos = transform.apply(pos);
        bool result = createPoint(str_id, transformed_pos, flags, button_flags, marker_radius, marker_col);
        if (ImGui::IsItemActivated())
            saveActivatedPoint();
        return result;
    }

    bool ImControlContext::controlPoint(const char* str_id, const Transformations::TransformationStack& transform, const ImVec4& pos, ImControlPointFlags flags, ImGuiButtonFlags button_flags, float marker_radius, ImVec4 marker_col)
    {
        IM_ASSERT(m_free_parameters.size() > 0);  // Else nothing to change, may relax this in future

        ImVec4 transformed_pos = transform.apply(pos);

        bool result = createPoint(str_id, transformed_pos, flags, button_flags, marker_radius, marker_col);

        if (ImGui::IsItemActive() && !ImGui::IsItemActivated())
        {
            // Derivatives with respect to the parameters are required to move the control point in the next frame
            for (auto param : m_free_parameters) {
                registerFreeParameterDerivativeForNextFrame(param);
                registerFreeParameterSecondDerForNextFrame(param);
            }
       
            ImVec2 mp = getMousePositionInViewCoords();
            ImVec4 mouse_in_view_coords{ mp[0], mp[1], transformed_pos.z / transformed_pos.w, 1.0f };

            // Could add this alternative behaviour behind a flag
            //ImVec4 mouse_in_view_coords{ mp[0], mp[1], m_activated_point_z_order, 1.0f };

            // Used to change the parameters
            ImVector<float> step = bring_together(transform, pos, mouse_in_view_coords, m_free_parameters, m_regularisation_constant);

            constexpr float softening_param = 1.0f;
            ImVector<parameter_change_t> changes{};
            for (int i = 0; i < m_free_parameters.size(); ++i)  // Zip the free parameters and their changes
                changes.push_back({ m_free_parameters[i], -softening_param * step[i] });

            updateParameters(changes, flags);

            if (flags & ImControlPointFlags_DrawParamDerivatives)
            {
                // Draw tangent vector(s)
                for (auto p : m_free_parameters) {
                    ImVec4 tangent = calculate_tangent_after_projection(transform, pos, p);
                    transformed_pos /= transformed_pos.w;
                    tangent -= transformed_pos * tangent.w;  // Required in unusual case when w-coord is non-zero
                    drawDerivative({ transformed_pos.x, transformed_pos.y }, { tangent.x, tangent.y }, marker_col);
                }
            }
        }
        else if (ImGui::IsItemActivated()) 
        {
            saveActivatedPoint(m_free_parameters);
        }
        return result;
    }

    ImControlContext g_current_context{};
    Transformations::TransformationStack g_current_transform{};

    void NewFrame() {
        g_current_context.endFrame();
        g_current_context.newFrame(g_current_transform);
    }

    bool BeginView(const char* name, ImVec2 const& size, ImVec4 const& border_color, bool defer_changes)
    {
        return g_current_context.beginView(name, size, border_color, defer_changes);
    }

    void EndView()
    {
        g_current_context.endView(false);
    }

    bool EndViewAsButton(ImGuiWindowFlags flags)
    {
        return g_current_context.endView(true, flags);
    }

    void PushDeferralSlot()
    {
        g_current_context.pushDeferralSlot();
    }

    void PopDeferralSlot()
    {
        g_current_context.popDeferralSlot();
    }

    void BeginCompositeTransformation()
    {
        g_current_transform.pushCompositeLevel();
    }

    void EndCompositeTransformation()
    {
        g_current_transform.popCompositeLevel();
    }

    ImVec2 GetViewBoundsMin()
    {
        return g_current_context.getViewBoundsMin();
    }

    ImVec2 GetViewBoundsMax()
    {
        return g_current_context.getViewBoundsMax();
    }

    ImVec2 GetViewSize()
    {
        return g_current_context.getViewSize();
    }

    float GetViewAspectRatio()
    {
        auto& size = g_current_context.getViewSize();
        return size.x / size.y;
    }

    ImVec2 GetLastControlPointPosition() {
        return g_current_context.getLastControlPointPosition();
    }

    float GetLastControlPointRadius() {
        return g_current_context.getLastControlPointRadius();
    }

    void SetRegularisationParameter(float kappa)
    {
        g_current_context.setRegularisationParam(kappa);
    }

    ImVec4 ApplyTransformation(ImVec4 const& pos)
    {
        return g_current_transform.apply(pos);
    }
    
    ImVec2 ApplyTransformation(ImVec2 const& pos)
    {
        ImVec4 embedded_pos{ pos.x, pos.y, 0.0f, 1.0f };
        ImVec4 result = ApplyTransformation(embedded_pos);
        return ImVec2{ result.x / result.w, result.y / result.w };
    }

    ImVec2 ApplyTransformationScreenCoords(ImVec2 const& pos)
    {
        return g_current_context.pointInScreenCoords(g_current_transform, { pos.x, pos.y, 0.0f, 1.0f });
    }

    ImVec2 ApplyTransformationScreenCoords(ImVec4 const& pos)
    {
        return g_current_context.pointInScreenCoords(g_current_transform, pos);
    }

    void PushFreeParameter(float* param)
    {
        g_current_context.pushFreeParameter(param);
    }

    void PopFreeParameter()
    {
        g_current_context.popFreeParameter();
    }

    void RestrictParameter(float* param)
    {
        g_current_context.restrictParameter(param);
    }

    void ClearFreeParameters()
    {
        g_current_context.clearParameters();
    }

    bool Button(const char* str_id, ImVec2 const& pos, ImControlPointFlags flags, ImGuiButtonFlags button_flags, float marker_radius, ImVec4 marker_col, float z_order)
    {
        return g_current_context.staticControlPoint(str_id, g_current_transform, { pos.x, pos.y, z_order, 1 }, flags, button_flags, marker_radius, marker_col);
    }

    bool Button(const char* str_id, ImVec4 const& pos, ImControlPointFlags flags, ImGuiButtonFlags button_flags, float marker_radius, ImVec4 marker_col)
    {
        return g_current_context.staticControlPoint(str_id, g_current_transform, pos, flags, button_flags, marker_radius, marker_col);
    }

    bool Point(const char* str_id, ImVec2 const& pos, ImControlPointFlags flags, ImGuiButtonFlags button_flags, float marker_radius, ImVec4 marker_col, float z_order)
    {
        return g_current_context.controlPoint(str_id, g_current_transform, { pos.x, pos.y, z_order, 1 }, flags, button_flags, marker_radius, marker_col);
    }

    bool Point(const char* str_id, ImVec4 const& pos, ImControlPointFlags flags, ImGuiButtonFlags button_flags, float marker_radius, ImVec4 marker_col)
    {
        return g_current_context.controlPoint(str_id, g_current_transform, pos, flags, button_flags, marker_radius, marker_col);
    }

    void PushConstantMatrix(ImMat4 const& M) {
        g_current_transform.pushConstantTransformation(Transformations::ConstantMatrix{ M });
    }

    void PushConstantPerspectiveMatrix(float fov, float aspect_ratio, float z_near, float z_far) {
        float tan_half_fov = tanf(fov / 2);
        PushConstantMatrix(ImMat4{
            ImVec4{1 / (aspect_ratio * tan_half_fov), 0, 0, 0},
            ImVec4{0, 1 / tan_half_fov, 0, 0},
            ImVec4{0, 0, -(z_near + z_far) / (z_far - z_near), -1},
            ImVec4{0, 0, -2 * z_far * z_near / (z_far - z_near), 0}
            });
    }

    void PushConstantLookAtMatrix(ImVec4 const& eye, ImVec4 const& center, ImVec4 const& up) {
        IM_ASSERT(center.w != 0.0f); IM_ASSERT(eye.w != 0.0f); IM_ASSERT(up.w == 0.0f);
        ImVec4 const f{ normalise(center / center.w - eye / eye.w) };
        ImVec4 const s{ normalise(cross(f, up)) };
        ImVec4 const u{ cross(s, f) };
        PushConstantMatrix(ImMat4{
            ImVec4{ s.x, u.x, -f.x, 0 },
            ImVec4{ s.y, u.y, -f.y, 0 },
            ImVec4{ s.z, u.z, -f.z, 0 },
            ImVec4{-dot(s, eye), -dot(u, eye), dot(f, eye), 1.0f }
            });
    }

    void PushRotationAboutAxis(float* parameter, ImVec4 const& axis) { g_current_transform.pushTransformation(Transformations::RotationAroundAxis{ axis, *parameter }, parameter); }
    void PushConstantRotationAboutAxis(float angle, ImVec4 const& axis) {
        Transformations::RotationAroundAxis T{ axis, angle };
        g_current_transform.pushConstantTransformation(T);
    }
    void PushRotationAboutX(float* parameter) { g_current_transform.pushTransformation(Transformations::RotationX{ *parameter }, parameter); }
    void PushRotationAboutY(float* parameter) { g_current_transform.pushTransformation(Transformations::RotationY{ *parameter }, parameter); }
    void PushRotationAboutZ(float* parameter) { g_current_transform.pushTransformation(Transformations::RotationZ{ *parameter }, parameter); }
    void PushRotationAboutX(const Parameters::Parameter& param) { g_current_transform.pushTransformation(Transformations::RotationX{ param.value }, param); }
    void PushRotationAboutY(const Parameters::Parameter& param) { g_current_transform.pushTransformation(Transformations::RotationY{ param.value }, param); }
    void PushRotationAboutZ(const Parameters::Parameter& param) { g_current_transform.pushTransformation(Transformations::RotationZ{ param.value }, param); }
    void PushConstantRotationAboutX(float angle) { g_current_transform.pushConstantTransformation(Transformations::RotationX{ angle }); }
    void PushConstantRotationAboutY(float angle) { g_current_transform.pushConstantTransformation(Transformations::RotationY{ angle }); }
    void PushConstantRotationAboutZ(float angle) { g_current_transform.pushConstantTransformation(Transformations::RotationZ{ angle }); }

    void PushRotation(float* parameter) { PushRotationAboutZ(parameter); }
    void PushRotation(const Parameters::Parameter& param) { PushRotationAboutZ(param); }
    void PushConstantRotation(float angle) { PushConstantRotationAboutZ(angle); }

    void PushScale(float* parameter) { g_current_transform.pushTransformation(Transformations::Scale{ *parameter }, parameter); }
    void PushScaleX(float* parameter) { g_current_transform.pushTransformation(Transformations::ScaleCoordinate<0>{ *parameter }, parameter); }
    void PushScaleY(float* parameter) { g_current_transform.pushTransformation(Transformations::ScaleCoordinate<1>{ *parameter }, parameter); }
    void PushScaleZ(float* parameter) { g_current_transform.pushTransformation(Transformations::ScaleCoordinate<2>{ *parameter }, parameter); }
    void PushScaleAlongAxis(float* parameter, const ImVec4& axis) { g_current_transform.pushTransformation(Transformations::ScaleInDirection{ *parameter, axis }, parameter); }
    void PushScaleFromAxis(float* parameter, const ImVec4& axis) { g_current_transform.pushTransformation(Transformations::ScaleAboutAxis{ *parameter, axis }, parameter); }
    void PushScale(const Parameters::Parameter& param) { g_current_transform.pushTransformation(Transformations::Scale{ param.value }, param); }
    void PushScaleX(const Parameters::Parameter& param) { g_current_transform.pushTransformation(Transformations::ScaleCoordinate<0>{ param.value }, param); }
    void PushScaleY(const Parameters::Parameter& param) { g_current_transform.pushTransformation(Transformations::ScaleCoordinate<1>{ param.value }, param); }
    void PushScaleZ(const Parameters::Parameter& param) { g_current_transform.pushTransformation(Transformations::ScaleCoordinate<2>{ param.value }, param); }
    void PushScaleAlongAxis(const Parameters::Parameter& param, const ImVec4& axis) { g_current_transform.pushTransformation(Transformations::ScaleInDirection{ param.value, axis }, param); }
    void PushScaleFromAxis(const Parameters::Parameter& param, const ImVec4& axis) { g_current_transform.pushTransformation(Transformations::ScaleAboutAxis{ param.value, axis }, param); }
    void PushConstantScale(float factor) { g_current_transform.pushConstantTransformation(Transformations::Scale{ factor }); }
    void PushConstantScaleX(float factor) { g_current_transform.pushConstantTransformation(Transformations::ScaleCoordinate<0>{ factor }); }
    void PushConstantScaleY(float factor) { g_current_transform.pushConstantTransformation(Transformations::ScaleCoordinate<1>{ factor }); }
    void PushConstantScaleZ(float factor) { g_current_transform.pushConstantTransformation(Transformations::ScaleCoordinate<2>{ factor }); }
    void PushConstantScaleAlongAxis(float factor, const ImVec4& axis) { g_current_transform.pushConstantTransformation(Transformations::ScaleInDirection{ factor, axis }); }
    void PushConstantScaleFromAxis(float factor, const ImVec4& axis) { g_current_transform.pushConstantTransformation(Transformations::ScaleAboutAxis{ factor, axis }); }
    void PushConstantReflection(const ImVec4& normal) { PushConstantScaleAlongAxis(-1.0f, normal); }

    void PushTranslation(float* parameter, ImVec4 const& v) { g_current_transform.pushTransformation(Transformations::Translation{ v, *parameter }, parameter); }
    void PushTranslation(float* parameter, ImVec2 const& v) { g_current_transform.pushTransformation(Transformations::Translation{ { v.x, v.y, 0.0f, 0.0f }, *parameter }, parameter); }
    void PushTranslation(const Parameters::Parameter& param, ImVec4 const& v) { g_current_transform.pushTransformation(Transformations::Translation{ v, param.value }, param); }
    void PushTranslation(const Parameters::Parameter& param, ImVec2 const& v) { g_current_transform.pushTransformation(Transformations::Translation{ { v.x, v.y, 0.0f, 0.0f }, param.value }, param); }
    void PushConstantTranslation(float value, ImVec4 const& v) { g_current_transform.pushConstantTransformation(Transformations::Translation{ v, value }); }
    void PushConstantTranslation(float value, ImVec2 const& v) { g_current_transform.pushConstantTransformation(Transformations::Translation{ { v.x, v.y, 0.0f, 0.0f }, value }); }

    void PushTranslationAlongX(float* parameter) { g_current_transform.pushTransformation(Transformations::TranslationOfCoordinate<0>{ *parameter }, parameter); }
    void PushTranslationAlongY(float* parameter) { g_current_transform.pushTransformation(Transformations::TranslationOfCoordinate<1>{ *parameter }, parameter); }
    void PushTranslationAlongZ(float* parameter) { g_current_transform.pushTransformation(Transformations::TranslationOfCoordinate<2>{ *parameter }, parameter); }
    void PushTranslationAlongX(const Parameters::Parameter& param) { g_current_transform.pushTransformation(Transformations::TranslationOfCoordinate<0>{ param.value }, param); }
    void PushTranslationAlongY(const Parameters::Parameter& param) { g_current_transform.pushTransformation(Transformations::TranslationOfCoordinate<1>{ param.value }, param); }
    void PushTranslationAlongZ(const Parameters::Parameter& param) { g_current_transform.pushTransformation(Transformations::TranslationOfCoordinate<2>{ param.value }, param); }
    void PushConstantTranslationAlongX(float value) { g_current_transform.pushConstantTransformation(Transformations::TranslationOfCoordinate<0>{ value }); }
    void PushConstantTranslationAlongY(float value) { g_current_transform.pushConstantTransformation(Transformations::TranslationOfCoordinate<1>{ value }); }
    void PushConstantTranslationAlongZ(float value) { g_current_transform.pushConstantTransformation(Transformations::TranslationOfCoordinate<2>{ value }); }

    void PopTransformation() { g_current_transform.popTransformation(); }
    void ClearTransformations() { g_current_transform.clear(); }

    // Composite transformations
    void PushTranslation(float* x, float* y)
    {
        BeginCompositeTransformation();
        PushTranslationAlongX(x);
        PushTranslationAlongY(y);
        EndCompositeTransformation();
    }

    void PushTranslation(float* x, float* y, float* z) {
        BeginCompositeTransformation();
        PushTranslationAlongX(x);
        PushTranslationAlongY(y);
        PushTranslationAlongZ(z);
        EndCompositeTransformation();
    }

    void PushRotationAboutPoint(float* angle, const ImVec2& p)
    {
        BeginCompositeTransformation();
        PushConstantTranslation(-1.0f, p);
        PushRotation(angle);
        PushConstantTranslation(1.0f, p);
        EndCompositeTransformation();
    }

    void PushScaleXY(float* factor)
    {
        BeginCompositeTransformation();
        PushScaleX(factor);
        PushScaleY(factor);
        EndCompositeTransformation();
    }

    ImMat4 GetTransformationMatrix()
    {
        return g_current_transform.getMatrix();
    }

    int GetTransformationStackWidth()
    {
        return static_cast<int>(g_current_transform.getStackWidth());
    }

    ImVec4 GetDerivativeAt(ImVec4 const& v, float* parameter)
    {
        g_current_context.registerFreeParameterDerivativeForNextFrame(parameter);
        return g_current_transform.applyDerivative(v, parameter);
    }

    ImVec4 GetSecondDerivativeAt(ImVec4 const& v, float* p1, float* p2)
    {
        g_current_context.registerFreeParameterSecondDerForNextFrame(p1);
        g_current_context.registerFreeParameterSecondDerForNextFrame(p2);
        return g_current_transform.applySecondDerivative(v, p1, p2);
    }

    void ParameterChangeDeferralStack::addParameterChange(ImVector<parameter_change_t> const& changes)
    {
        IM_ASSERT(m_deferral_stack_size > 0);  // There should be a slot in which to place our deferred changes
        m_deferred_changes = changes;
        m_deferred_change_position = m_deferral_stack_size - 1;  // Step is at the end of the stack
    }
    
    void ParameterChangeDeferralStack::reset()
    {
        m_deferral_stack_size = 1;
        m_deferred_change_position = -1;
        m_deferred_changes = {};
    }
    
    bool ParameterChangeDeferralStack::applyParameterChanges()
    {
        bool has_changed = false;
    
        for (auto const& change : m_deferred_changes) {
            if (change.change && change.parameter)
            {
                *(change.parameter) += change.change;
                has_changed = true;
            }
        }
        m_deferred_change_position = -1;
        m_deferred_changes = {};
        return has_changed;
    }
    
    void ParameterChangeDeferralStack::pushDeferralSlot()
    {
        ++m_deferral_stack_size;
    }
    
    bool ParameterChangeDeferralStack::popDeferralSlot()
    {
        IM_ASSERT(m_deferral_stack_size > 0);  // We shouldn't pop an empty stack
    
        bool change_applied = false;
    
        // If there is a change at the top of the stack then apply it
        if (m_deferred_change_position + 1 == m_deferral_stack_size)
            change_applied = applyParameterChanges();
    
        // Reduce the size
        --m_deferral_stack_size;
        return change_applied;
    }
}  // end namespace ImControl