#include <chrono>

#include "imgui.h"
#include "imcontrol.h"
#include "imcontrol_example_widgets.h"

namespace ImControl {

    // Individual demo windows
    void ShowTreeDemo(bool*);
    void ShowCameraDemo(bool*);
    void ShowSpiralDemo(bool*);
    void ShowArmDemo(bool*);
    void ShowDraggableShapesDemo(bool*);
    void ShowGaussianDemo(bool*);
    void ShowKnotDemo(bool*);
    void ShowWidgetDemo(bool*);

    static void HelpMarker(const char* desc)
    {
        ImGui::TextDisabled("(?)");
        if (ImGui::IsItemHovered())
        {
            ImGui::BeginTooltip();
            ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
            ImGui::TextUnformatted(desc);
            ImGui::PopTextWrapPos();
            ImGui::EndTooltip();
        }
    }

    void ShowDemoWindow(bool* p_open)
    {
        static bool show_arm_demo = false;
        static bool show_draggable_shapes_demo = false;
        static bool show_camera_demo = false;
        static bool show_gaussian_demo = false;
        static bool show_knot_demo = false;
        static bool show_spiral_demo = false;
        static bool show_tree_demo = false;
        static bool show_widget_demo = false;

        static bool show_tests = false;
        static bool show_benchmarks = false;


        ImGui::Begin("Control Point Demos", p_open);

        ImGui::Checkbox("show arm demo window", &show_arm_demo);
        ImGui::Checkbox("show shapes demo window", &show_draggable_shapes_demo);
        ImGui::Checkbox("show knot demo window", &show_knot_demo);
        ImGui::Checkbox("show Gaussian demo window", &show_gaussian_demo);
        //ImGui::Checkbox("show spiral demo window", &show_spiral_demo);
        ImGui::Checkbox("show tree demo window", &show_tree_demo);
        //ImGui::Checkbox("show camera demo window", &show_camera_demo);
        ImGui::Checkbox("show widget window", &show_widget_demo);

        ImGui::Separator();
        ImGui::Checkbox("show tests window", &show_tests);
        ImGui::Checkbox("show benchmarks window", &show_benchmarks);

        ImGui::Separator();
        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);

        ImGui::Separator();
        ImGui::Text("Parameters");

        static float kappa = 0.5f;
        ImGui::SliderFloat("regularisation", &kappa, 0.0001f, 2.0f);
        ImControl::SetRegularisationParameter(kappa);

        ImGui::End();

        if (show_spiral_demo)
            ShowSpiralDemo(&show_spiral_demo);

        if (show_tree_demo)
            ShowTreeDemo(&show_tree_demo);

        if (show_gaussian_demo)
            ShowGaussianDemo(&show_gaussian_demo);

        if (show_knot_demo)
            ShowKnotDemo(&show_knot_demo);

        if (show_camera_demo)
            ShowCameraDemo(&show_camera_demo);

        if (show_arm_demo)
            ShowArmDemo(&show_arm_demo);

        if (show_draggable_shapes_demo)
            ShowDraggableShapesDemo(&show_draggable_shapes_demo);

        if (show_widget_demo)
            ShowWidgetDemo(&show_widget_demo);

        if (show_tests)
            ShowTestsWindow(&show_tests);

        if (show_benchmarks)
            ShowBenchmarkWindow(&show_benchmarks);
    }

    inline float min(float a, float b) { return a < b ? a : b; }
    inline float max(float a, float b) { return a > b ? a : b; }
    inline float clamp(float x, float a, float b) { return min(max(a, x), b); }

    struct TreeParameters {
        float main_length;
        float left_angle;
        float left_log_scale_factor;
        float right_angle;
        float right_log_scale_factor;
    };

    using namespace ImControl::Parameters;

    // A custom transformation made up of a composite of basic transformations
    void RotateAndExpScale(float* angle, float* log_scaling_factor)
    {
        ImControl::BeginCompositeTransformation();
        ImControl::PushRotation(angle);
        ImControl::PushScaleX(Exponential(Parameter{ log_scaling_factor }));
        ImControl::PushScaleY(Exponential(Parameter{ log_scaling_factor }));
        ImControl::EndCompositeTransformation();
    }

    void create_tree(TreeParameters* params, int depth, int LR, ImControlPointFlags flags)
    {
        if (depth <= 0)
            return;

        // Current branch
        ImVec2 start_point = ImControl::ApplyTransformationScreenCoords({ 0.0f, 0.0f });
        ImControl::PushTranslationAlongY(&(params->main_length));
        if (LR == 0) {
            //ImControl::PushFreeParameter(&params->main_length);
            ImControl::Point("", { 0.0f, 0.0f }, flags, 0, 0.025f, { 0, 0, 1, 1 }, 1.0f / depth);
            //ImControl::PopFreeParameter();
        }
        else if (LR == 1) { // Right hand branch
            //ImControl::PushFreeParameter(&params->right_log_scale_factor);
            //ImControl::PushFreeParameter(&params->right_angle);
            ImControl::Point("", { 0.0f, 0.0f }, flags, 0, 0.025f, { 0, 1, 0, 1 }, 1.0f / depth);
            //ImControl::PopFreeParameter();
            //ImControl::PopFreeParameter();
        }
        else if (LR == -1) { // Left hand branch
            //ImControl::PushFreeParameter(&params->left_log_scale_factor);
            //ImControl::PushFreeParameter(&params->left_angle);
            ImControl::Point("", { 0.0f, 0.0f }, flags, 0, 0.025f, { 1, 0, 0, 1 }, 1.0f / depth);
            //ImControl::PopFreeParameter();
            //ImControl::PopFreeParameter();
        }
        ImVec2 end_point = ImControl::ApplyTransformationScreenCoords({ 0.0f, 0.0f });

        // Draw the branch
        ImGui::GetWindowDrawList()->AddLine(start_point, end_point, ImGui::GetColorU32({ 1.0, 1.0, 0.0, 1.0 }), 1.0f);

        // Left branch
        RotateAndExpScale(&(params->left_angle), &(params->left_log_scale_factor));
        ImGui::PushID("L");
        create_tree(params, depth - 1, -1, flags);
        ImGui::PopID();
        ImControl::PopTransformation();

        // Right branch
        RotateAndExpScale(&(params->right_angle), &(params->right_log_scale_factor));
        ImGui::PushID("R");
        create_tree(params, depth - 1, 1, flags);
        ImGui::PopID();
        ImControl::PopTransformation();

        ImControl::PopTransformation();  // Translation along Y
    }

    // An example showing the use of the 2D transformation context, using an static instance of the Draggable2DView class
    void ShowTreeDemo(bool* p_open)
    {
        ImGui::Begin("Fractal Tree - using 2D transforms", p_open, ImGuiWindowFlags_NoScrollbar);

        static int depth = 7;
        static TreeParameters parameters{ 0.5f, -1.18f, -0.5f, 0.25f, -0.3f };
        static bool draw_markers = false;
        static bool draw_derivatives = false;

        static bool left_angle_free = true;
        static bool right_angle_free = true;
        static bool left_scale_free = true;
        static bool right_scale_free = true;
        static bool total_scale_free = false;

        float log_factor_lower_bound = logf(0.2f);
        float log_factor_upper_bound = 0;
        float angle_lower_bound = -3.1416f / 2;
        float angle_upper_bound = 3.1416f / 2;

        ImGui::Checkbox("##leftfactor", &left_scale_free); ImGui::SameLine();
        ImGui::SliderFloat("left branch log factor", &parameters.left_log_scale_factor, log_factor_lower_bound, log_factor_upper_bound);
        
        ImGui::Checkbox("##leftangle", &left_angle_free); ImGui::SameLine(); 
        ImGui::SliderFloat("left branch angle", &parameters.left_angle, angle_lower_bound, angle_upper_bound);
        
        ImGui::Checkbox("##rightscale", &right_scale_free); ImGui::SameLine();
        ImGui::SliderFloat("right branch log factor", &parameters.right_log_scale_factor, log_factor_lower_bound, log_factor_upper_bound);
        
        ImGui::Checkbox("##rightangle", &right_angle_free); ImGui::SameLine();
        ImGui::SliderFloat("right branch angle", &parameters.right_angle, angle_lower_bound, angle_upper_bound);
        
        ImGui::Checkbox("##totalscale", &total_scale_free); ImGui::SameLine();
        ImGui::SliderFloat("stem length", &parameters.main_length, 0.1f, 1.5f);
        
        bool dummybool{ false }; ImGui::Checkbox("##dummyfortree", &dummybool); ImGui::SameLine(); // for alignment only
        ImGui::SliderInt("tree depth", &depth, 1, 12);

        ImGui::Checkbox("draw control points", &draw_markers);
        ImGui::Checkbox("draw derivatives", &draw_derivatives);

        if (ImControl::BeginView("tree_view", ImVec2{ -1, -1 }, ImVec4(1, 1, 1, 1)))
        {
            ImControlPointFlags flags = 0;
            if (draw_markers)
                flags |= ImControlPointFlags_DrawControlPointMarkers;
            if (draw_derivatives)
                flags |= ImControlPointFlags_DrawParamDerivatives;

            ImControl::PushConstantScaleX(-1.0f / ImControl::GetViewAspectRatio());
            ImControl::PushConstantScaleY(-1.0f);
            ImControl::PushConstantTranslationAlongY(-0.8f);

            if(total_scale_free)
                ImControl::PushFreeParameter(&parameters.main_length);
            if(left_angle_free)
                ImControl::PushFreeParameter(&parameters.left_angle);
            if(left_scale_free)
                ImControl::PushFreeParameter(&parameters.left_log_scale_factor);
            if(right_angle_free)
                ImControl::PushFreeParameter(&parameters.right_angle);
            if(right_scale_free)
                ImControl::PushFreeParameter(&parameters.right_log_scale_factor);
            create_tree(&parameters, depth, 0, flags);
            ImControl::ClearFreeParameters();

            ImControl::ClearTransformations();

            ImControl::EndView();  // Any changes to parameters happen in this call

            // Enforce bounds on parameters should happen after parameters have been changed
            parameters.left_log_scale_factor = clamp(parameters.left_log_scale_factor, log_factor_lower_bound, log_factor_upper_bound);
            parameters.right_log_scale_factor = clamp(parameters.right_log_scale_factor, log_factor_lower_bound, log_factor_upper_bound);
            parameters.left_angle = clamp(parameters.left_angle, angle_lower_bound, angle_upper_bound);
            parameters.right_angle = clamp(parameters.right_angle, angle_lower_bound, angle_upper_bound);
        }
        ImGui::End();
    }


    void ShowSpiralDemo(bool* p_open)
    {
        ImGui::Begin("Spiral Demo", p_open);

        static float spiral_rotation = 2.05f;
        static float spiral_height = 0.025f;
        static float spiral_scale = 0.9f;
        static int spiral_length = 20;
        static bool draw_control_points = false;
        static bool draw_derivatives = false;

        ImGui::SliderFloat("spiral rotation", &spiral_rotation, -3.1416f, 3.1416f);
        ImGui::SliderFloat("spiral height", &spiral_height, -0.05f, 0.05f);
        ImGui::SliderFloat("spiral scale", &spiral_scale, 0.7f, 1.1f);
        ImGui::SliderInt("number of points", &spiral_length, 2, 100);
        ImGui::Checkbox("draw control points", &draw_control_points);
        ImGui::Checkbox("draw derivatives", &draw_derivatives);

        if (ImControl::BeginView("some_label_for_id", ImVec2(300, 300), ImVec4(0, 1, 1, 1)))
        {
            ImControl::BeginCompositeTransformation();
            ImControl::PushConstantPerspectiveMatrix(45.0f / 180.f * 3.14159f, 1.0f, 0.01f, 10.0f);
            ImControl::PushConstantLookAtMatrix(
                { 2.5f, -0.8f, 1.0f, 1.0f }, { 0.0f, 0.0f, 0.0f, 1.0f }, { 0.0f, 0.0f, -1.0f, 0.0f }
            );

            auto draw_list = ImGui::GetWindowDrawList();
            ImControl::Button("start", { 1.0, 0.0, 0.0, 1.0 }, 0, 0, 0.05f, ImVec4{ 0, 1, 1, 1 });
            ImVec2 prev_point = ImControl::GetLastControlPointPosition();

            ImControlPointFlags flags = 0;
            if (draw_control_points)
                flags |= ImControlPointFlags_DrawControlPointMarkers;
            if (draw_derivatives)
                flags |= ImControlPointFlags_DrawParamDerivatives;

            ImControl::PushFreeParameter(&spiral_rotation);
            ImControl::PushFreeParameter(&spiral_scale);
            for (int i = 0; i < spiral_length; ++i)
            {
                ImControl::PushRotationAboutZ(&spiral_rotation);
                ImControl::PushTranslationAlongZ(&spiral_height);
                ImControl::PushScaleX(&spiral_scale);
                ImControl::PushScaleY(&spiral_scale);
                ImGui::PushID(i);
                ImControl::Point("", { 1.0, 0.0, 0.0, 1.0 }, flags, 0, 0.05f, ImVec4(1, 0, 0, 1));
                ImGui::PopID();
                ImVec2 current_point = ImControl::GetLastControlPointPosition();
                draw_list->PathLineTo(prev_point);
                draw_list->PathLineTo(current_point);
                draw_list->PathStroke(ImGui::GetColorU32({ 1.0, 0.0, 0.0, 1.0 }), false, 2.0f);
                prev_point = current_point;
            }
            ImControl::ClearFreeParameters();

            draw_list->PathStroke(ImGui::GetColorU32({ 1.0, 0.0, 0.0, 1.0 }), false);
            ImControl::PushConstantTranslationAlongZ(0.1f);
            ImControl::PushConstantScaleX(0.0f);
            ImControl::PushConstantScaleY(0.0f);
            ImControl::EndCompositeTransformation();

            flags |= ImControlPointFlags_DrawControlPointMarkers;
            ImControl::PushFreeParameter(&spiral_height);
            ImControl::Point("peak", { 1.0, 0.0, 0.0, 1.0 }, flags, 0, 0.05f, ImVec4(0, 1, 0, 1));
            ImControl::PopFreeParameter();
            ImControl::PopTransformation();
            ImControl::EndView();

            // Clamping of parameter changes should occur only after the incremental changes have been made, usually after EndView()
            spiral_scale = clamp(spiral_scale, 0.7f, 1.1f);
            spiral_height = clamp(spiral_height, -0.05f, 0.05f);
        }

        ImGui::End();
    }

    // These are defined in imcontrol_internal.h, but we don't want everything from there
    ImVec4 operator+ (const ImVec4& v, const ImVec4& w) { return ImVec4{ v.x + w.x, v.y + w.y, v.z + w.z, v.w + w.w }; }
    ImVec4 operator* (const ImVec4& v, float s) { return ImVec4{ v.x * s, v.y * s, v.z * s, v.w * s }; }
    ImVec4 operator* (const ImMat4& M, const ImVec4& v) { return M.col[0] * v.x + M.col[1] * v.y + M.col[2] * v.z + M.col[3] * v.w; }

    void draw_knot(int p, int q, ImMat4 P, int n_segments=300) 
    {
        auto m = ImControl::GetViewBoundsMin();
        auto M = ImControl::GetViewBoundsMax();
        auto view_to_screen_coords = [m, M](const ImVec2& p) { return ImVec2{ (M.x - m.x) * (p.x + 1) / 2 + m.x, (M.y - m.y) * (p.y + 1) / 2 + m.y }; };

        auto draw_list = ImGui::GetWindowDrawList();

        for (int i = 0; i < n_segments; ++i) {
            float phi = (i * 2 * 3.14159f) / n_segments;
            float r = cosf(q * phi) + 2;
            ImVec4 pos = P * ImVec4{ r * cosf(p * phi), r * sinf(p * phi), -sinf(q * phi), 1.0f };
            draw_list->PathLineTo(view_to_screen_coords({ pos.x / pos.w, pos.y / pos.w }));
        }
        draw_list->PathStroke(ImGui::GetColorU32({ 1, 1, 0, 1 }), true, 2.0f);
    }

    void ShowKnotDemo(bool* p_open)
    {
        ImGui::Begin("Knot Demo", p_open, ImGuiWindowFlags_NoScrollbar);
        ImGui::Text("Move a point on a (p, q)-torus knot");
        static int pq[2]{ 2, 3 };
        static bool draw_derivatives = false;
        static float bead_pos{};
        static float camera_angle{};
        static bool fixed_angle{};

        ImGui::SliderInt2("(p, q)", pq, 2, 10);
        ImGui::SliderFloat("view angle", &camera_angle, 0, 3.14159f / 2);
        ImGui::Checkbox("view fixed", &fixed_angle);
        ImGui::Checkbox("draw derivatives", &draw_derivatives);

        if (ImControl::BeginView("some_label_for_id", ImVec2(-1, -1), ImVec4(0, 1, 1, 1)))
        {
            ImControl::PushConstantPerspectiveMatrix(45.0f / 180.f * 3.14159f, ImControl::GetViewAspectRatio(), 0.01f, 10.0f);
            ImControl::PushConstantLookAtMatrix(
                { 0.0f, 0.0f, 9.0f, 1.0f }, { 0.0f, 0.0f, 0.0f, 1.0f }, { 0.0f, 1.0f, 0.0f, 0.0f }
            );
            ImControl::PushRotationAboutX(&camera_angle);

            draw_knot(pq[0], pq[1], ImControl::GetTransformationMatrix());

            auto pphi = Linear((float)pq[0], 0, Parameter{ &bead_pos });
            auto qphi = Linear((float)pq[1], 0, Parameter{ &bead_pos });
            auto r = Cosine(qphi) + 2;
            ImControl::BeginCompositeTransformation();
            ImControl::PushTranslationAlongX(r * Cosine(pphi));
            ImControl::PushTranslationAlongY(r * Sine(pphi));
            ImControl::PushTranslationAlongZ(Sine(qphi) * -1);
            ImControl::EndCompositeTransformation();

            ImControlPointFlags flags = ImControlPointFlags_DrawControlPointMarkers | ImControlPointFlags_Circular;
            if (draw_derivatives)
                flags |= ImControlPointFlags_DrawParamDerivatives;

            ImControl::PushFreeParameter(&bead_pos);
            if (!fixed_angle)
                ImControl::PushFreeParameter(&camera_angle);

            ImControl::Point("bead", { 0, 0, 0, 1 }, flags, 0, 0.25f, { 1, 1, 1, 1 });

            ImControl::ClearFreeParameters();

            ImControl::ClearTransformations();
            ImControl::EndView();
        }

        ImGui::End();
    }

    void ShowArmDemo(bool* p_open)
    {
        ImGui::Begin("Arm Demo", p_open, ImGuiWindowFlags_NoScrollbar);

        static float base_rotation = 0.0f;
        static float base_angle = 0.1f;
        static float elbow_angle = 0.2f;
        constexpr bool include_reflection = true;
        float point_alpha = 1.0f;

        ImControlPointFlags flags = ImControlPointFlags_DrawControlPointMarkers | ImControlPointFlags_DrawParamDerivatives;

        // Begin the viewport, filling available space
        ImControl::BeginView("arm", { -1, -1 }, { 1, 1, 1, 1 });

        
        // Push projection and camera matrices
        ImControl::PushConstantPerspectiveMatrix(45.0f / 180.0f * 3.14159f, ImControl::GetViewAspectRatio(), 0.01f, 10.0f);
        ImControl::PushConstantLookAtMatrix({ 0.0f, 0.9f, -1.5f, 1.0f }, { 0, 0.3f, 0.15f, 1 }, { 0, -1, 0, 0 });

        for (int i = 0; i < 2; ++i)
        {
            ImGui::PushID(i);
            ImControl::BeginCompositeTransformation();
            ImControl::PushRotationAboutY(&base_rotation);  // Rotate arm

            ImControl::PushFreeParameter(&base_rotation);
            ImControl::Point("rotation", { 0.2f, 0, 0, 1.0f }, flags, 0, 0.0375f, { 1, 0, 1, point_alpha });  // Controls Rotation
            ImControl::PopFreeParameter();

            ImControl::Button("base", { 0.0f, 0, 0, 1.0f }, flags, 0, 0.025f, { 1, 1, 1, point_alpha });  // Demarks base of arm

            ImControl::PushRotationAboutZ(&base_angle);  // Rotate arm up and down
            ImControl::PushConstantTranslationAlongY(0.5f);  // Move up to next joint

            ImControl::PushFreeParameter(&base_angle);
            ImControl::Point("arm1", { 0.0f, -0.1f, 0, 1.0f }, flags, 0, 0.025f, { 1, 1, 1, point_alpha });  // Control points along arm
            ImControl::Point("arm2", { 0.0f, -0.2f, 0, 1.0f }, flags, 0, 0.025f, { 1, 1, 1, point_alpha });
            ImControl::Point("arm3", { 0.0f, -0.3f, 0, 1.0f }, flags, 0, 0.025f, { 1, 1, 1, point_alpha });
            ImControl::Point("arm4", { 0.0f, -0.4f, 0, 1.0f }, flags, 0, 0.025f, { 1, 1, 1, point_alpha });
            ImControl::Point("elbow", { 0.0f, 0, 0, 1.0f }, flags, 0, 0.0375f, { 1, 1, 1, point_alpha });  // Control point at joint
            ImControl::PopFreeParameter();

            ImControl::PushRotationAboutZ(&elbow_angle);  // Bend the elbow
            ImControl::PushConstantTranslationAlongY(0.4f);  // Move to end of arm
            ImControl::EndCompositeTransformation();  // Done transforming

            ImControl::PushFreeParameter(&elbow_angle);
            ImControl::Point("forearm1", { 0.0f, -0.1f, 0, 1.0f }, flags, 0, 0.025f, { 1, 1, 0, point_alpha });  // Control points along forearm
            ImControl::Point("forearm2", { 0.0f, -0.2f, 0, 1.0f }, flags, 0, 0.025f, { 1, 1, 0, point_alpha });
            ImControl::Point("forearm3", { 0.0f, -0.3f, 0, 1.0f }, flags, 0, 0.025f, { 1, 1, 0, point_alpha });
            ImControl::PushFreeParameter(&base_angle);
            ImControl::Point("hand", { 0.0f, 0, 0, 1.0f }, flags, 0, 0.0375f, { 1, 0, 0, point_alpha });  // Hand controls two angles
            ImControl::ClearFreeParameters();

            ImControl::PopTransformation();
            ImGui::PopID();

            if (!include_reflection)
                continue;

            if (i == 0) {
                ImControl::PushConstantReflection({ -0.3f, 0, 1, 0 });
                ImControl::PushConstantTranslationAlongZ(-1.0f);  // Together these are a reflection in the plane z = 0.5f
                point_alpha = 0.8f;
            }
        }

        ImControl::ClearTransformations();
        ImControl::EndView();  // Parameter changes are made here

        elbow_angle = clamp(elbow_angle, 0.3f, 2.8f);  // Clamp angles after view ends
        base_angle = clamp(base_angle, 0.1f, 1.4f);

        ImGui::End();
    }

    void draw_circle(ImMat4 P, float r = 1.0f, int n_segments = 40) {
        auto m = ImControl::GetViewBoundsMin();
        auto M = ImControl::GetViewBoundsMax();
        auto view_to_screen_coords = [m, M](const ImVec2& p) { return ImVec2{ (M.x - m.x) * (p.x + 1) / 2 + m.x, (M.y - m.y) * (p.y + 1) / 2 + m.y }; };
        auto world_to_view_coords = [P](const ImVec2& p) { auto q = P * ImVec4{ p.x, p.y, 0.0f, 1.0f }; return ImVec2{ q.x / q.w, q.y / q.w }; };

        auto draw_list = ImGui::GetWindowDrawList();

        for (int i = 0; i < n_segments; ++i) {
            float angle = (2 * 3.14159f * (i + 0.5f)) / n_segments;
            draw_list->PathLineTo(view_to_screen_coords(world_to_view_coords({ r * cosf(angle), r * sinf(angle) })));
        }
        draw_list->PathStroke(ImGui::GetColorU32({ 1, 1, 0, 1 }), true, 2.0f);
    }

    void draw_square(ImMat4 P) {
        draw_circle(P, sqrtf(2.0f), 4);
    }

    void ShowDraggableShapesDemo(bool* p_open)
    {
        ImGui::Begin("Draggable Shapes Demo", p_open, ImGuiWindowFlags_NoScrollbar);

        ImGui::Text("Click and drag the control points."); ImGui::Separator();
        ImGui::TextWrapped("Note that the circle is quick to rotate, the ellipse and rectangle less so, and the square reluctant.  This is controlled by reparametrisations of the rotation variables, and this technique can give you more control of the behaviour of control points.  You may also wish to use alter the global regularisation parameter to see how this affects the dragging behaviour (see the slider in the demo window).");

        // Variables defining the state of the shapes
        static float circle_rotation{};
        static ImVec2 circle_center{ 1.1f, -1.1f };

        static float square_rotation{};
        static ImVec2 square_center{ 1.1f, 1.5f };

        static float rect_rotation{};
        static ImVec2 rect_center{ -1.5f, 1.2f };
        static ImVec2 rect_scale{ 1.1f, 0.9f };

        static float ellipse_rotation{};
        static ImVec2 ellipse_center{ -1.1f, -0.1f };

        ImControlPointFlags flags = ImControlPointFlags_DrawControlPointMarkers | ImControlPointFlags_DrawParamDerivatives;

        // Begin the viewport, filling available space
        ImControl::BeginView("arm", { -1, -1 }, { 1, 1, 1, 1 });

        // Push projection and camera matrices
        ImControl::PushConstantPerspectiveMatrix(45.0f / 180.0f * 3.14159f, ImControl::GetViewAspectRatio(), 0.01f, 10.0f);
        ImControl::PushConstantLookAtMatrix({ 0.0f, 4.0f, 4.0f, 1.0f }, { 0, 0, 0, 1 }, { 0, 0, -1, 0 });

        // The circle
        ImControl::PushTranslation(&circle_center[0], &circle_center[1]);
        ImControl::PushRotation(Linear(5, 0, Parameter{ &circle_rotation }));  // Reparametrisation makes the circle easier to turn
        draw_circle(ImControl::GetTransformationMatrix());

        ImControl::PushFreeParameter(&circle_rotation);
        ImControl::PushFreeParameter(&circle_center[0]);
        ImControl::PushFreeParameter(&circle_center[1]);
        constexpr int n_points = 40;
        for (int i = 0; i < n_points; ++i) {
            float angle = i * (2 * 3.14159f / n_points);
            ImGui::PushID(i);
            ImControl::Point("circle", { cosf(angle), sinf(angle), 0, 1 }, flags, 0, 0.1f, { 1, 1, 1, 1 });
            ImGui::PopID();
        }
        ImControl::ClearFreeParameters();
        
        ImControl::PopTransformation();
        ImControl::PopTransformation();


        // The square
        ImControl::PushTranslation(&square_center[0], &square_center[1]);
        ImControl::PushRotation(Linear(0.5f, 0, Parameter{ &square_rotation }));  // Reparametrisation makes the square harder to turn
        draw_square(ImControl::GetTransformationMatrix());

        ImControl::PushFreeParameter(&square_rotation);
        ImControl::PushFreeParameter(&square_center[0]);
        ImControl::PushFreeParameter(&square_center[1]);
        ImControl::Point("square1", {  1,  1, 0, 1 }, flags, 0, 0.1f, { 1, 1, 1, 1 });
        ImControl::Point("square2", {  1, -1, 0, 1 }, flags, 0, 0.1f, { 1, 1, 1, 1 });
        ImControl::Point("square3", { -1, -1, 0, 1 }, flags, 0, 0.1f, { 1, 1, 1, 1 });
        ImControl::Point("square4", { -1,  1, 0, 1 }, flags, 0, 0.1f, { 1, 1, 1, 1 });
        ImControl::ClearFreeParameters();
        ImControl::PopTransformation();
        ImControl::PopTransformation();


        // The rectangle
        ImControl::BeginCompositeTransformation();
        ImControl::PushTranslation(&rect_center[0], &rect_center[1]);
        ImControl::PushRotation(&rect_rotation);
        ImControl::PushScaleX(&rect_scale[0]);
        ImControl::PushScaleY(&rect_scale[1]);
        ImControl::EndCompositeTransformation();

        draw_square(ImControl::GetTransformationMatrix());

        ImControl::PushFreeParameter(&rect_scale[0]);
        ImControl::PushFreeParameter(&rect_scale[1]);
        ImControl::Point("rect1", { 1,  1, 0, 1 }, flags, 0, 0.1f, { 1, 1, 1, 1 });
        ImControl::Point("rect2", { 1, -1, 0, 1 }, flags, 0, 0.1f, { 1, 1, 1, 1 });
        ImControl::Point("rect3", { -1, -1, 0, 1 }, flags, 0, 0.1f, { 1, 1, 1, 1 });
        ImControl::Point("rect4", { -1,  1, 0, 1 }, flags, 0, 0.1f, { 1, 1, 1, 1 });
        ImControl::ClearFreeParameters();

        ImControl::PushFreeParameter(&rect_rotation);
        ImControl::PushFreeParameter(&rect_scale[0]);
        ImControl::Point("side1", {  1, 0, 0, 1 }, flags, 0, 0.1f, { 1, 0, 1, 1 });
        ImControl::Point("side2", { -1, 0, 0, 1 }, flags, 0, 0.1f, { 1, 0, 1, 1 });
        ImControl::PopFreeParameter();
        ImControl::PushFreeParameter(&rect_scale[1]);
        ImControl::Point("side3", { 0,  1, 0, 1 }, flags, 0, 0.1f, { 0, 1, 1, 1 });
        ImControl::Point("side4", { 0, -1, 0, 1 }, flags, 0, 0.1f, { 0, 1, 1, 1 });
        ImControl::ClearFreeParameters();

        ImControl::PushFreeParameter(&rect_center[0]);
        ImControl::PushFreeParameter(&rect_center[1]);
        ImControl::Point("rect_center", {  0, 0, 0, 1 }, flags, 0, 0.1f, { 1, 1, 0, 1 });
        ImControl::ClearFreeParameters();

        ImControl::PopTransformation();  // Only one pop as we use a composite above


        // The ellipse
        ImControl::BeginCompositeTransformation();
        ImControl::PushTranslation(&ellipse_center[0], &ellipse_center[1]);
        ImControl::PushRotation(&ellipse_rotation);
        ImControl::PushConstantScaleX(0.6f);
        ImControl::PushConstantScaleY(1 / 0.6f);
        ImControl::EndCompositeTransformation();
        draw_circle(ImControl::GetTransformationMatrix());

        ImControl::PushFreeParameter(&ellipse_rotation);
        ImControl::PushFreeParameter(&ellipse_center[0]);
        ImControl::PushFreeParameter(&ellipse_center[1]);

        for (int i = 0; i < n_points; ++i) {
            float angle = i * (2 * 3.14159f / n_points);
            ImGui::PushID(i);
            ImControl::Point("ellipse", { cosf(angle), sinf(angle), 0, 1 }, flags, 0, 0.1f, { 1, 1, 1, 1 });
            ImGui::PopID();
        }
        ImControl::ClearFreeParameters();
        
        ImControl::PopTransformation();

        ImControl::ClearTransformations();
        ImControl::EndView();  // Parameter changes are made here

        ImGui::End();
    }

    static inline float unit_gaussian(float x) { return 1.0f / (sqrtf(2 * 3.14159f)) * expf(-0.5f * x * x); }

    void draw_gaussian(float amp, float mean, float stddev, const ImVec2& scale, const ImVec2& offset, int n_segments = 200)
    {
        auto m = ImControl::GetViewBoundsMin();
        auto M = ImControl::GetViewBoundsMax();
        auto view_to_screen_coords = [m, M](const ImVec2& p) { return ImVec2{ (M.x - m.x) * (p.x + 1) / 2 + m.x, (M.y - m.y) * (p.y + 1) / 2 + m.y }; };

        auto cartesian_to_view_coords = [scale, offset](const ImVec2& p) { return ImVec2{ (p.x - offset.x) / scale.x, (p.y - offset.y) / scale.y }; };
        auto view_to_cartesian_coords = [scale, offset](const ImVec2& p) { return ImVec2{ scale.x * p.x + offset.x, scale.y * p.y + offset.y }; };

        auto f = [amp, mean, stddev](float x) { return amp / stddev * unit_gaussian((x - mean) / stddev); };

        auto draw_list = ImGui::GetWindowDrawList();

        // Draw axes
        auto view_axis_center = cartesian_to_view_coords({ 0, 0 });
        draw_list->AddLine(view_to_screen_coords({ -1, view_axis_center.y }), view_to_screen_coords({ 1, view_axis_center.y }), ImGui::GetColorU32({ 1, 1, 1, .8f }));
        draw_list->AddLine(view_to_screen_coords({ view_axis_center.x, view_axis_center.y }), view_to_screen_coords({ view_axis_center.x, -1 }), ImGui::GetColorU32({ 1, 1, 1, .5f }));

        // Draw Gaussian curve
        //draw_list->PathLineTo(view_to_screen_coords({ -1.0f, f(-1.0f) }));
        for (int i = 0; i <= n_segments; ++i) {
            float view_x = (2.0f * i) / n_segments - 1.0f;
            float x = view_to_cartesian_coords({ view_x, 0 }).x;
            draw_list->PathLineTo(view_to_screen_coords(cartesian_to_view_coords({ x, f(x) })));
        }
        draw_list->PathStroke(ImGui::GetColorU32({ 1, 1, 0, 1 }), false);
    }


    void ShowGaussianDemo(bool* p_open)
    {
        static float peak = 1.2f;
        static float mu = 0.25f;
        static float sigma = 0.5f;

        static bool show_debug_line = false;
        static bool show_markers = false;
        static bool show_derivatives = false;
        static bool alternative_method = false;

        ImGui::Begin("Gaussian chooser widget", p_open);
        ImGui::TextWrapped("Click and drag the curve to change its parameters.");

        float ampl = peak * sqrtf(2 * 3.14159f) * sigma;
        if (ImGui::SliderFloat("amplitude", &ampl, 0.0f, 5.0f))
            peak = ampl / (sqrtf(2 * 3.14159f) * sigma);

        ImGui::SliderFloat("mean", &mu, -2.0f, 2.0f);
        if(ImGui::SliderFloat("std dev.", &sigma, 0.01f, 2.0f, "%.3f", ImGuiSliderFlags_Logarithmic))
            peak = ampl / (sqrtf(2 * 3.14159f) * sigma);

        ImGui::SliderFloat("peak height", &peak, 0.1f, 4.0f);
        ImGui::SameLine(); HelpMarker("When dragging the curve it is the mean, std. dev. and peak height which are edited, the amplitude is then calculated from these.  Changing the amplitude directly is problematic as there is a singularity when the standard deviative is small.  One could hide this slider and the user would not need to consider this.");

        ImGui::Checkbox("show debug line", &show_debug_line); ImGui::SameLine(); HelpMarker("See the code that generates this line for a helpful debug code-snippet");
        ImGui::Checkbox("draw markers", &show_markers);
        ImGui::Checkbox("draw derivatives", &show_derivatives);
        ImGui::Checkbox("alternative method", &alternative_method); ImGui::SameLine(); HelpMarker("Drag the peak to change the mean and amplitude, drag the sides where they are steepest to change the standard deviation and amplitude.");

        ImControlPointFlags flags = ImControlPointFlags_Circular | ImControlPointFlags_ChooseClosestWhenOverlapping | ImControlPointFlags_SizeInPixels;
        if (show_markers)
            flags |= ImControlPointFlags_DrawControlPointMarkers;
        if (show_derivatives)
            flags |= ImControlPointFlags_DrawParamDerivatives;

        ImControl::BeginView("Gaussian function", { -1, 200 }, { 1, 1, 1, 1 });
        auto view_size = ImControl::GetViewSize();
        float aspect_ratio = view_size.x / view_size.y;

        draw_gaussian(ampl, mu, sigma, { aspect_ratio, -1.0 }, { 0.0f, 0.75f });

        ImControl::BeginCompositeTransformation();
        ImControl::PushConstantTranslationAlongY(0.75f);
        ImControl::PushConstantScaleX(1.0f / aspect_ratio);
        ImControl::PushConstantScaleY(-1.0f);
        ImControl::PushScaleY(&peak);
        ImControl::PushTranslationAlongX(&mu);
        ImControl::PushScaleX(&sigma);
        ImControl::EndCompositeTransformation();

        ImControl::PushFreeParameter(&peak);
        ImControl::PushFreeParameter(&sigma);
        if (!alternative_method)  // Control all three at once, unless alternative_method
            ImControl::PushFreeParameter(&mu);
        
        constexpr float marker_size = 7.0f;
        for (int i = -35; i <= 35; ++i) {
            float x = 0.05f * i;
            if (alternative_method && (i == -9)) {
                ImControl::PopFreeParameter();  // Change parameter over to mu
                ImControl::PushFreeParameter(&mu);
            }
            else if (alternative_method && (i == 10)) {
                ImControl::PopFreeParameter();  // Change parameter back to sigma again
                ImControl::PushFreeParameter(&sigma);
            }
            ImGui::PushID(i);
            ImControl::Point("curve", ImVec2{ x, expf(-0.5f * x * x) }, flags, 0, marker_size, { 1, 0, 0, 1 }, 0);
            ImGui::PopID();
        }
        ImControl::ClearFreeParameters();

        if (show_debug_line) {  // This code snippet is very useful for tracking down out-of-bound control points
            ImVec2 pos = ImControl::GetLastControlPointPosition();
            ImGui::GetForegroundDrawList()->AddLine(pos, { 100, 100 }, ImGui::GetColorU32({ 1, 0, 0, 1 }));
        }

        ImControl::PopTransformation();
        ImControl::EndView();
        ImGui::End();

        // Bounding variables should occur after the parameter has changed, in this case after EndView
        peak = clamp(peak, 0.0f, 20.0f);
        sigma = clamp(sigma, 0.001f, 4.0f);
        mu = clamp(mu, -4, 4);
    }

    // Our prototype translation widget, the transform stack / control point syntax can be used to create more complicated control widgets
    void ShowWidgetDemo(bool* p_open)
    {
        static bool rotate_widget = false;
        static ImMat4 model_matrix{
            ImVec4{1, 0, 0, 0},
            ImVec4{0, 1, 0, 0},
            ImVec4{0, 0, 1, 0},
            ImVec4{0, 0, 0, 1}
        };

        ImGui::Begin("Translation/Rotation Widget", p_open, ImGuiWindowFlags_NoScrollbar);
        ImGui::TextWrapped("Click in the view to switch between rotation and translation.");

        static bool use_circular_markers{ true };
        ImGui::Checkbox("circular markers", &use_circular_markers);

        auto size = ImGui::GetContentRegionAvail().x;

        if (ImControl::BeginView("widget", { size, size }, { 1, 1, 1, 1 }))
        {
            ImControl::PushConstantPerspectiveMatrix(45.0f / 180.0f * 3.14159f, 1.0f, 0.01f, 10.0f);
            ImControl::PushConstantLookAtMatrix({ 0.0f, 0.25f, -1.0f, 1.0f }, { 0, 0, 0, 1 }, { 0, -1, 0, 0 });

            // Draw our control points
            ImControlPointFlags flags = ImControlPointFlags_DrawControlPointMarkers;
            if (use_circular_markers)
                flags |= ImControlPointFlags_Circular;

            ImControl::Button("center", model_matrix[3], flags, 0, 0.03f, { 1, 1, 1, 1 });
            if(rotate_widget)
                CustomWidgets::RotationWidget(&model_matrix, 0.1f, flags);
            else
                CustomWidgets::TranslationWidget(&model_matrix, 0.1f, flags, false);

            ImControl::PopTransformation();
            ImControl::PopTransformation();
            if (ImControl::EndViewAsButton())
                rotate_widget = !rotate_widget;
        }
        if (ImGui::Button("Reset Position"))
            model_matrix = { ImVec4{1, 0, 0, 0}, ImVec4{0, 1, 0, 0}, ImVec4{0, 0, 1, 0}, ImVec4{ 0, 0, 0, 1 } };

        ImGui::End();
    }

    void ShowCameraDemo(bool* p_open)
    {
        ImGui::Begin("Camera Demo", p_open, ImGuiWindowFlags_NoScrollbar);
    
        ImGui::Text("Work in progress");
    
        static float cube_rotation = 0.0f;
        static float camera_vertical_angle = 0.0f;
        static float camera_distance = 3.0f;
        static bool show_parameter_derivatives = false;
    
        ImGui::SliderFloat("cube rotation", &cube_rotation, -10.0f, 10.0f);
        ImGui::SliderFloat("vertical angle", &camera_vertical_angle, -3.1416f / 4, 3.1416f / 4);
        ImGui::SliderFloat("camera displacement", &camera_distance, -5.0f, 5.0f);
        ImGui::Checkbox("show parameter derivatives", &show_parameter_derivatives);
    
        ImControlPointFlags flags = ImControlPointFlags_DrawControlPointMarkers;
        if (show_parameter_derivatives)
            flags |= ImControlPointFlags_DrawParamDerivatives;
    
        ImControl::BeginView("camera", ImVec2(-1, -1), ImVec4(0, 1, 1, 1));

        // Setup transformations
        ImControl::BeginCompositeTransformation();
        ImControl::PushConstantPerspectiveMatrix(45.0f * 3.14159f / 180.f, ImControl::GetViewAspectRatio(), 0.01f, 10.0f);  // projection matrix
        ImControl::PushTranslationAlongZ(Linear(-1.0f, 0.0f, Parameter{ &camera_distance }));  // move camera back


        ImControl::PushRotationAboutX(&camera_vertical_angle);
        ImControl::PushRotationAboutY(&cube_rotation);  // Rotation of model along long axis (y is up)
        ImControl::PushConstantLookAtMatrix(  // Model matrix, centering cube at origin and standing it on its diagonal axis
            { 0.5f, 0.5f, 0.5f, 1.0f }, { 4.5f, -3.0f, 0.0f, 1.0f }, { -1.0f, -1.0f, -1.0f, 0.0f }
        );
        ImControl::EndCompositeTransformation();
    
        // Control points at vertices of cube
        ImControl::PushFreeParameter(&cube_rotation);
        ImControl::Point("110", { 1.0f, 1.0f, 0.0f, 1.0f }, flags, 0, 0.05f, { 1, 0, 0, 1 });
        ImControl::Point("101", { 1.0f, 0.0f, 1.0f, 1.0f }, flags, 0, 0.05f, { 1, 0, 0, 1 });
        ImControl::Point("011", { 0.0f, 1.0f, 1.0f, 1.0f }, flags, 0, 0.05f, { 1, 0, 0, 1 });
        ImControl::PopFreeParameter();

        ImControl::PushFreeParameter(&camera_vertical_angle);
        ImControl::Point("100", { 1.0f, 0.0f, 0.0f, 1.0f }, flags, 0, 0.05f, { 1, 0, 0, 1 });
        ImControl::Point("010", { 0.0f, 1.0f, 0.0f, 1.0f }, flags, 0, 0.05f, { 1, 0, 0, 1 });
        ImControl::Point("001", { 0.0f, 0.0f, 1.0f, 1.0f }, flags, 0, 0.05f, { 1, 0, 0, 1 });
        ImControl::PopFreeParameter();

        ImControl::PushFreeParameter(&camera_distance);
        ImControl::Point("111", { 1.0f, 1.0f, 1.0f, 1.0f }, flags, 0, 0.05f, { 1, 1, 1, 1 });
        ImControl::PopFreeParameter();

        ImControl::Button("000", { 0.0f, 0.0f, 0.0f, 1.0f }, flags, 0, 0.05f, { 1, 1, 0, 1 });

        ImControl::PopTransformation();
        ImControl::EndView();

    }

    // Defined for the following tests
    static ImVec4 operator-(ImVec4 const& v, ImVec4 const& w) { return { v.x - w.x, v.y - w.y, v.z - w.z, v.w - w.w }; }
    static ImVec4 operator/(ImVec4 const& v, float d) { return { v.x / d, v.y / d, v.z / d, v.w / d }; }
    static float dot(ImVec4 const& v, ImVec4 const& w) { return v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w; }
    static float length(ImVec4 const& v) { return dot(v, v); }
    static void output_vector(ImVec4 const& v) { ImGui::Text("(%.2f, %.2f, %.2f, %.2f)", v.x, v.y, v.z, v.w); }
    static void output_vector(const char* text, ImVec4 const& v) { ImGui::Text(text); ImGui::SameLine(); output_vector(v); }

    static void comparison(ImVec4 const& d, ImVec4 const& est_d, ImVec4 const& est_2nd_d, ImVec4 const& sd, float eps, float threshold) {
        output_vector("Estimated derivative:", est_d);
        output_vector("Calculated derivative:", d);
        float a = length(est_d - d) / (eps * eps);
        ImGui::TextColored((a < threshold) ? ImVec4{ 0, 1, 0, 1 } : ImVec4{ 1, 0, 0, 1 }, "Difference / epsilon^2: %.2f", a);

        output_vector("Estimated second derivative:", est_2nd_d);
        output_vector("Calculated second derivative:", sd);
    }

    void ShowTestsWindow(bool* p_open)
    {
        ImGui::Begin("Tests", p_open);

        ImGui::Text("Calculated derivatives vs numerical approximations");
        static float epsilon = 0.005f;
        static float param = 0.2f;
        static float param_plus_eps;
        ImGui::SliderFloat("parameter", &param, 0.0f, 1.0f, "%.4f");
        ImGui::SliderFloat("epsilon", &epsilon, 0.0001f, 1.0f, "%.4f", ImGuiSliderFlags_Logarithmic);
        param_plus_eps = epsilon + param;
        ImVec4 v{ 2, 1, 3, 1 };  // Some point

        ImGui::Separator();

        // Tests of individual transformations
        ImVec4 w{}, derivative{}, estimated_derivative{}, second_derivative, estimated_second_derivative;

        ImGui::Text("Rotation about x-axis");
        ImControl::PushRotationAboutX(&param);
        w = ImControl::ApplyTransformation(v);
        derivative = ImControl::GetDerivativeAt(v, &param);
        ImControl::PopTransformation();

        ImControl::PushRotationAboutX(&param_plus_eps);
        estimated_derivative = (ImControl::ApplyTransformation(v) - w) / epsilon;
        second_derivative = ImControl::GetSecondDerivativeAt(v, &param_plus_eps, &param_plus_eps);
        estimated_second_derivative = (ImControl::GetDerivativeAt(v, &param_plus_eps) - derivative) / epsilon;
        ImControl::PopTransformation();

        comparison(derivative, estimated_derivative, second_derivative, estimated_second_derivative, epsilon, 10.0f);

        ImGui::Separator();
        ImGui::Text("Rotation about axis (1, 1, 1)");
        ImControl::PushRotationAboutAxis(&param, { 1, 1, 1, 0 });
        w = ImControl::ApplyTransformation(v);
        derivative = ImControl::GetDerivativeAt(v, &param);
        ImControl::PopTransformation();

        ImControl::PushRotationAboutAxis(&param_plus_eps, { 1, 1, 1, 0 });
        estimated_derivative = (ImControl::ApplyTransformation(v) - w) / epsilon;
        second_derivative = ImControl::GetSecondDerivativeAt(v, &param_plus_eps, &param_plus_eps);
        estimated_second_derivative = (ImControl::GetDerivativeAt(v, &param_plus_eps) - derivative) / epsilon;
        ImControl::PopTransformation();

        comparison(derivative, estimated_derivative, second_derivative, estimated_second_derivative, epsilon, 10.0f);

        ImGui::Separator();
        ImGui::Text("Translation along y-axis");
        ImControl::PushTranslationAlongY(&param);
        w = ImControl::ApplyTransformation(v);
        derivative = ImControl::GetDerivativeAt(v, &param);
        ImControl::PopTransformation();

        ImControl::PushTranslationAlongY(&param_plus_eps);
        estimated_derivative = (ImControl::ApplyTransformation(v) - w) / epsilon;
        second_derivative = ImControl::GetSecondDerivativeAt(v, &param_plus_eps, &param_plus_eps);
        estimated_second_derivative = (ImControl::GetDerivativeAt(v, &param_plus_eps) - derivative) / epsilon;
        ImControl::PopTransformation();

        comparison(derivative, estimated_derivative, second_derivative, estimated_second_derivative, epsilon, 10.0f);

        ImGui::Separator();
        ImGui::Text("Scale along z-axis");
        ImControl::PushScaleZ(&param);
        w = ImControl::ApplyTransformation(v);
        derivative = ImControl::GetDerivativeAt(v, &param);
        ImControl::PopTransformation();

        ImControl::PushScaleZ(&param_plus_eps);
        estimated_derivative = (ImControl::ApplyTransformation(v) - w) / epsilon;
        second_derivative = ImControl::GetSecondDerivativeAt(v, &param_plus_eps, &param_plus_eps);
        estimated_second_derivative = (ImControl::GetDerivativeAt(v, &param_plus_eps) - derivative) / epsilon;
        ImControl::PopTransformation();

        comparison(derivative, estimated_derivative, second_derivative, estimated_second_derivative, epsilon, 10.0f);

        // Test for a composition of a number of transformations
        static float p2 = -0.5f;
        ImGui::Separator();
        ImGui::Text("Composite transformation");

        ImControl::PushScale(&param);
        ImControl::PushTranslation(&p2, { 1, 2, 3, 0 });
        ImControl::PushConstantPerspectiveMatrix(0.5f, 1.0f, 0.1f, 10.0f);
        ImControl::PushScaleY(Cosine(Exponential(Parameter{ &param })));

        w = ImControl::ApplyTransformation(v);
        derivative = ImControl::GetDerivativeAt(v, &param);
        ImVec4 deriv_p2 = ImControl::GetDerivativeAt(v, &p2);
        ImVec4 calc_sd2 = ImControl::GetSecondDerivativeAt(v, &p2, &param);
        ImControl::ClearTransformations();

        ImControl::PushScale(&param_plus_eps);
        ImControl::PushTranslation(&p2, { 1, 2, 3, 0 });
        ImControl::PushConstantPerspectiveMatrix(0.5f, 1.0f, 0.1f, 10.0f);
        ImControl::PushScaleY(Cosine(Exponential(Parameter{ &param_plus_eps })));

        estimated_derivative = (ImControl::ApplyTransformation(v) - w) / epsilon;
        second_derivative = ImControl::GetSecondDerivativeAt(v, &param_plus_eps, &param_plus_eps);
        estimated_second_derivative = (ImControl::GetDerivativeAt(v, &param_plus_eps) - derivative) / epsilon;
        ImVec4 est_sd2 = (ImControl::GetDerivativeAt(v, &p2) - deriv_p2) / epsilon;
        ImControl::ClearTransformations();

        comparison(derivative, estimated_derivative, second_derivative, estimated_second_derivative, epsilon, 50.0f);  // Larger threshold as a composite
        output_vector("Estimated second derivative (offdiagonal):", est_sd2);
        output_vector("Calculated second derivative (offdiagonal):", calc_sd2);

        ImGui::Separator();

        ImGui::End();
    }

    template<typename T>
    inline T max(const ImVector<T>& v) {
        T max{};
        for (const auto& w : v)
            max = (w > max) ? w : max;
        return max;
    }

    class BenchmarkedValue {
        ImVector<float> history{};
        int history_ix{};
        bool first_save{ true };
        float smoothed_value{};
        float max_value{};

    public:
        const int history_size{ 1 };
        float smoothing{ 0.99f };

        BenchmarkedValue(int size) : history_size{ size } {
            history.resize(size);
        };
        float saveValue(float value) {
            if (first_save) {
                smoothed_value = value;
                max_value = value;
                first_save = false;
            }

            history[history_ix] = value;
            if (++history_ix == history_size) history_ix = 0;

            smoothed_value = smoothing * smoothed_value + (1 - smoothing) * value;

            max_value = 0.9f * max_value + 0.1f * max(history);
            return smoothed_value;
        }
        float getSmoothedValue() const { return smoothed_value; }
        void plotValues(const char* label, ImVec2 plot_size = { 0, 0 }) {
            ImGui::PlotLines(label, history.Data, history_size, 0, nullptr, 0, max_value, plot_size);
        }
        void doSomeWork(void (*f)(ImVector<float>&, int), ImVector<float>& parameters, int n) {
            auto t_start = std::chrono::high_resolution_clock::now();
            f(parameters, n);
            auto t_end = std::chrono::high_resolution_clock::now();

            auto duration = (t_end - t_start).count();
            saveValue((float)duration / n);
        }
    };


    void push_some_rotations(ImVector<float>& parameters, int num_transformations) {
        float* p = parameters.begin();
        for (int i = 0; i < num_transformations; i += 3) {
            if (++p == parameters.end())
                p = parameters.begin();
            ImControl::PushRotationAboutX(p);
            ImControl::PushRotationAboutY(p);
            ImControl::PushRotationAboutZ(p);
        }
    }

    void push_some_matrices(ImVector<float>& parameters, int num_transformations) {
        ImMat4 M{};
        for (int i = 0; i < num_transformations; ++i) {
            M[0].x += 0.001f;  // Make some minor changes so our compiler won't do anything too unexpected (probably not needed)
            ImControl::PushConstantMatrix(M);
        }
    }

    void push_some_translations(ImVector<float>& parameters, int num_transformations) {
        float* p = parameters.begin();
        for (int i = 0; i < num_transformations; i += 3) {
            if (++p == parameters.end())
                p = parameters.begin();
            ImControl::PushTranslationAlongX(p);
            ImControl::PushTranslationAlongY(p);
            ImControl::PushTranslationAlongZ(p);
        }
    }

    void push_some_scalings(ImVector<float>& parameters, int num_transformations) {
        float* p = parameters.begin();
        for (int i = 0; i < num_transformations; i += 3) {
            if (++p == parameters.end())
                p = parameters.begin();
            ImControl::PushScaleX(p);
            ImControl::PushScaleY(p);
            ImControl::PushScaleZ(p);
            //ImControl::PushScale(p);
        }
    }

    void ShowBenchmarkWindow(bool* p_open)
    {

        static bool record_data = true;
        static bool show_plots = false;
        static float num_transformations_float = 1000;  // Number of transformations to perform
        constexpr int max_derivatives = 20;
        static int num_derivatives = 5;  // Number of derivatives to process
        static float factor = 0.9f;
        static float dist = 0.01f;
        static ImVector<float> parameters{};
        if (parameters.empty()) parameters.resize(num_derivatives + 1, 0.2f);

        ImGui::Begin("Benchmarks", p_open);
        ImGui::TextWrapped("These benchmarks shouldn't be taken too seriously, running the tests in a different order can give different results.");
        if (ImControl::GetTransformationStackWidth() != 1 + num_derivatives)
            ImGui::TextColored({ 1, 0, 0, 1 }, "Warning, there are extra derivatives in the stack.");
        ImGui::Checkbox("Record data", &record_data);
        ImGui::Checkbox("Show plots", &show_plots);
        ImGui::SliderFloat("Number of transformations", &num_transformations_float, 5, 2000, "%.0f", ImGuiSliderFlags_Logarithmic);
        int num_transformations = static_cast<int>(num_transformations_float);  // Workaround for logarithmic flag non-interoperable with a SliderInt
        if (ImGui::SliderInt("Number of derivative parameters", &num_derivatives, 0, max_derivatives, "%d", ImGuiSliderFlags_AlwaysClamp)) {
            parameters.resize(num_derivatives + 1, 0.2f);
        }

        constexpr int history_size = 500;
        static BenchmarkedValue rotations{ history_size };
        static BenchmarkedValue rotations_in_place{ history_size };
        static BenchmarkedValue apply_matrices{ history_size };
        static BenchmarkedValue apply_matrices_in_place{ history_size };
        static BenchmarkedValue translations{ history_size };
        static BenchmarkedValue translations_in_place{ history_size };
        static BenchmarkedValue scalings{ history_size };
        static BenchmarkedValue scalings_in_place{ history_size };

        if (record_data) {
            // Get derivatives for each parameter (skipping the first) so they'll be computed in the transformation stack
            for (float* p = parameters.begin() + 1; p != parameters.end(); ++p)
                ImControl::GetDerivativeAt({ 0, 0, 0, 1 }, p);

            for (int i = 0; i < 5; ++i)  // Performing multiple repeats interleaves the computations (the ordering makes a significant difference, perhaps due to branch prediction)
            {
                rotations.doSomeWork(&push_some_rotations, parameters, num_transformations);
                ImControl::ClearTransformations();
                apply_matrices.doSomeWork(&push_some_matrices, parameters, num_transformations);
                ImControl::ClearTransformations();
                translations.doSomeWork(&push_some_translations, parameters, num_transformations);
                ImControl::ClearTransformations();
                scalings.doSomeWork(&push_some_scalings, parameters, num_transformations);
                ImControl::ClearTransformations();

                ImControl::BeginCompositeTransformation();
                rotations_in_place.doSomeWork(&push_some_rotations, parameters, num_transformations);
                apply_matrices_in_place.doSomeWork(&push_some_matrices, parameters, num_transformations);
                translations_in_place.doSomeWork(&push_some_translations, parameters, num_transformations);
                scalings_in_place.doSomeWork(&push_some_scalings, parameters, num_transformations);
                ImControl::EndCompositeTransformation();
                ImControl::ClearTransformations();
            }
        }
        ImGui::Text("Taking %.1fns per rotation", rotations.getSmoothedValue());
        if (show_plots) rotations.plotValues("Rotations");
        ImGui::Text("Taking %.1fns per rotation in place", rotations_in_place.getSmoothedValue());
        if (show_plots) rotations_in_place.plotValues("Rotations (in place)");

        ImGui::Text("Taking %.1fns per matrix", apply_matrices.getSmoothedValue());
        if (show_plots) apply_matrices.plotValues("Matrices");
        ImGui::Text("Taking %.1fns per matrix in place", apply_matrices_in_place.getSmoothedValue());
        if (show_plots) apply_matrices_in_place.plotValues("Matrices (in place)");

        ImGui::Text("Taking %.1fns per translation", translations.getSmoothedValue());
        if (show_plots) translations.plotValues("Translations");
        ImGui::Text("Taking %.1fns per translation in place", translations_in_place.getSmoothedValue());
        if (show_plots) translations_in_place.plotValues("Translations (in place)");

        ImGui::Text("Taking %.1fns per scaling", scalings.getSmoothedValue());
        if (show_plots) scalings.plotValues("Scalings");
        ImGui::Text("Taking %.1fns per scaling in place", scalings_in_place.getSmoothedValue());
        if (show_plots) scalings_in_place.plotValues("Scalings (in place)");

        ImGui::Separator();
        ImGui::TextWrapped("We compare against pushing a scaling transformation.");
        ImGui::SameLine(); HelpMarker("We choose a scaling as it is the simplest and should be most stable under optimisation choices.");

        if (ImGui::BeginTable("comparisons", 5)) {
            float denom = scalings.getSmoothedValue();
            ImGui::TableSetupColumn("Comparisons");
            ImGui::TableSetupColumn("Scalings");
            ImGui::TableSetupColumn("Translations");
            ImGui::TableSetupColumn("Rotations");
            ImGui::TableSetupColumn("Matrices");
            ImGui::TableHeadersRow();

            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0); ImGui::Text("Simple push");
            ImGui::TableNextColumn(); ImGui::Text("1.0");
            ImGui::TableNextColumn(); ImGui::Text("%.2f", translations.getSmoothedValue() / denom);
            ImGui::TableNextColumn(); ImGui::Text("%.2f", rotations.getSmoothedValue() / denom);
            ImGui::TableNextColumn(); ImGui::Text("%.2f", apply_matrices.getSmoothedValue() / denom);

            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0); ImGui::Text("In-place push");
            ImGui::TableNextColumn(); ImGui::Text("%.2f", scalings_in_place.getSmoothedValue() / denom);
            ImGui::TableNextColumn(); ImGui::Text("%.2f", translations_in_place.getSmoothedValue() / denom);
            ImGui::TableNextColumn(); ImGui::Text("%.2f", rotations_in_place.getSmoothedValue() / denom);
            ImGui::TableNextColumn(); ImGui::Text("%.2f", apply_matrices_in_place.getSmoothedValue() / denom);

            ImGui::EndTable();
        }

        ImGui::End();
    }

}  // namespace ImControl