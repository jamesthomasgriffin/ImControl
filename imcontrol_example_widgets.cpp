#include "imcontrol_example_widgets.h"

#include "imcontrol_internal.h"

inline static void PushFreeParameters(float* p1, float* p2) { ImControl::PushFreeParameter(p1); ImControl::PushFreeParameter(p2); }
inline static void ReplaceFreeParameter(float* p) { ImControl::PopFreeParameter(); ImControl::PushFreeParameter(p); }
inline static void ReplaceFreeParameters(float* p1, float* p2) { ImControl::PopFreeParameter(); ImControl::PopFreeParameter(); ImControl::PushFreeParameter(p1); ImControl::PushFreeParameter(p2); }

namespace CustomWidgets
{
    void TranslationWidget(ImMat4* M, float axes_size, ImControlPointFlags control_point_flags, bool include_planes)
    {
        static ImVec4 t{};  // The translation vector (only three coords are used)
        t = {};  // set to zero each frame

        ImControl::BeginCompositeTransformation();
        ImControl::PushConstantMatrix(*M);
        ImControl::PushTranslation(&t.x, &t.y, &t.z);
        ImControl::PushConstantScale(axes_size);
        ImControl::EndCompositeTransformation();

        // We don't change any parameters at the control points or at the end of a view, instead when the deferral slot is popped
        ImControl::PushDeferralSlot();

        //ImControl::Button(ImVec4{ 0, 0, 0, 1 }, control_point_flags, 0, 0.025f, { 1, 1, 1, 1 });

        ImControl::PushFreeParameter(&t.x);
        ImControl::Point("x-axis", ImVec4{ 1, 0, 0, 1 }, control_point_flags, 0, 0.025f, { 1, 0, 0, 1 });
        ReplaceFreeParameter(&t.y);
        ImControl::Point("y-axis", ImVec4{ 0, 1, 0, 1 }, control_point_flags, 0, 0.025f, { 0, 1, 0, 1 });
        ReplaceFreeParameter(&t.z);
        ImControl::Point("z-axis", ImVec4{ 0, 0, 1, 1 }, control_point_flags, 0, 0.025f, { 0, 0, 1, 1 });
        ImControl::PopFreeParameter();

        if (include_planes) {
            PushFreeParameters(&t.x, &t.y);
            ImControl::Point("xy-plane", ImVec4{ 0.5f, 0.5f, 0, 1 }, control_point_flags, 0, 0.02f, { 1, 1, 0, 1 });
            ReplaceFreeParameters(&t.x, &t.y);
            ImControl::Point("yz-plane", ImVec4{ 0, 0.5f, 0.5f, 1 }, control_point_flags, 0, 0.02f, { 0, 1, 1, 1 });
            ReplaceFreeParameters(&t.x, &t.y);
            ImControl::Point("zx-plane", ImVec4{ 0.5f, 0, 0.5f, 1 }, control_point_flags, 0, 0.02f, { 1, 0, 1, 1 });
            ImControl::PopFreeParameter(); ImControl::PopFreeParameter();
        }

        // This will apply any changes made to t as intended, without the unintended consequence of
        // applying any previously made changes.
        ImControl::PopDeferralSlot();

        ImControl::PopTransformation();  // composite

        // Update the matrix if any of our parameters are non-zero
        M->col[3] += *M * t;
    }


    void RotationWidget(ImMat4* M, float widget_size, ImControlPointFlags control_point_flags)
    {
        static ImVec4 r{};  // The translation vector (only three coords are used)
        r = {};  // set to zero each frame

        ImControl::BeginCompositeTransformation();
        ImControl::PushConstantMatrix(*M);
        ImControl::PushRotationAboutX(ImControl::Parameters::Linear(10.0f, 0, ImControl::Parameters::Parameter{ &r.x }));  // the reparametrisation is to make the widget more responsive with default regularisation parameter
        ImControl::PushRotationAboutY(ImControl::Parameters::Linear(10.0f, 0, ImControl::Parameters::Parameter{ &r.y }));
        ImControl::PushRotationAboutZ(ImControl::Parameters::Linear(10.0f, 0, ImControl::Parameters::Parameter{ &r.z }));
        ImControl::PushConstantScale(widget_size);
        ImControl::EndCompositeTransformation();

        // We don't change any parameters at the control points or at the end of a view, instead when the deferral slot is popped
        ImControl::PushDeferralSlot();

        //ImControl::Button(ImVec4{ 0, 0, 0, 1 }, control_point_flags, 0, 0.02f, { 1, 1, 1, 1 });

        constexpr float marker_size = 0.015f;
        constexpr int n_markers = 48;
        for (int i = 0; i < n_markers; ++i) {
            ImGui::PushID(i);
            float angle = 2 * 3.14159f * i / n_markers;
            float a = cosf(angle), b = sinf(angle);
            ImControl::PushFreeParameter(&r.x);
            ImControl::Point("rotate-x", ImVec4{ 0, a, b, 1 }, control_point_flags, 0, marker_size, { 1, 0, 0, 1 });
            ReplaceFreeParameter(&r.y);
            ImControl::Point("rotate-y", ImVec4{ a, 0, b, 1 }, control_point_flags, 0, marker_size, { 0, 1, 0, 1 });
            ReplaceFreeParameter(&r.z);
            ImControl::Point("rotate-z", ImVec4{ a, b, 0, 1 }, control_point_flags, 0, marker_size, { 0, 0, 1, 1 });
            ImControl::PopFreeParameter();
            ImGui::PopID();
        }

        // This will apply any changes made to r as intended, without the unintended consequence of
        // applying any previously made changes.
        ImControl::PopDeferralSlot();

        ImControl::PopTransformation();  // all transformations

        // Update the matrix if any of our parameters are non-zero
        if (r.x != 0.0f) {
            ImControl::Transformations::RotationX T{ r.x };
            T.applyTransformationOnRightInPlace(M, M+1);
        }
        if (r.y != 0.0f) {
            ImControl::Transformations::RotationY T{ r.y };
            T.applyTransformationOnRightInPlace(M, M+1);
        }
        if (r.z != 0.0f) {
            ImControl::Transformations::RotationZ T{ r.z };
            T.applyTransformationOnRightInPlace(M, M+1);
        }
    }
}  // namespace CustomWidgets