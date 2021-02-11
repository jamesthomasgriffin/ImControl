#pragma once

// Include(s)
#include "math.h"
#include "imgui.h"

// Version
#define IMCONTROL_VERSION "0.1"

// Matrix classes - required operations defined in imcontrol_internal.h
struct ImMat4 {
	ImVec4 col[4];
	inline ImVec4& operator[](unsigned int i) { return col[i]; }
	inline const ImVec4& operator[](unsigned int i) const { return col[i]; }
};

struct ImMat2 {
	ImVec2 col[2];
};

typedef int ImControlPointFlags;

enum ImControlPointFlags_
{
	ImControlPointFlags_DrawControlPointMarkers = 1 << 0,
	ImControlPointFlags_DrawParamDerivatives = 1 << 1,
	ImControlPointFlags_DoNotChangeParams = 1 << 2,
	ImControlPointFlags_ApplyParamChangesImmediately = 1 << 3,
	ImControlPointFlags_Circular = 1 << 4,
	ImControlPointFlags_FixedSize = 1 << 5,
	ImControlPointFlags_SizeInPixels = 1 << 6,
	ImControlPointFlags_ChooseClosestWhenOverlapping = 1 << 7,
};


namespace ImControl {

	namespace Parameters {
		// For reparametrisation of transformations we require more than the
		// value of the reparametrisation, we also need the first and second
		// derivatives, this class holds exactly the information we require.
		// The usual arithmetic operations apply with multiplication given by
		// the product rule.
		class Parameter {
		public:
			explicit Parameter(float* param) : p{ param }, value{ *param }, d{ 1.0f }, d2{ 0.0f } {}
			Parameter(float* param, float val, float derivative, float second_derivative) : p{ param }, value{ val }, d{ derivative }, d2{ second_derivative } {}

			Parameter operator+(float b) const { return { p, value + b, d, d2 }; }
			Parameter operator*(float a) const { return { p, value * a, d * a, d2 * a }; }
			Parameter operator/(float a) const { return { p, value / a, d / a, d2 / a }; }
			Parameter operator+(const Parameters::Parameter& g) const { IM_ASSERT(p == g.p); return { p, value + g.value, d + g.d, d2 + g.d2 }; }
			Parameter operator-(const Parameters::Parameter& g) const { IM_ASSERT(p == g.p); return { p, value - g.value, d - g.d, d2 - g.d2 }; }
			Parameter operator*(const Parameters::Parameter& g) const { IM_ASSERT(p == g.p); return { p, value * g.value, d * g.value + value * g.d, d2 * g.value + 2 * d * g.d + value * g.d2 }; }  // Product rule
			Parameter reciprocal() const { return { p, 1 / value, -d / (value * value), (-d2 * value + 2 * d * d) / (value * value * value) }; }
			Parameter operator/(const Parameters::Parameter& g) const { IM_ASSERT(p == g.p); return *this * g.reciprocal(); }

			float value;    // value

			// These member variables may change to allow for multi-variate parameters - beware
			float* p;
			float d;   // derivative
			float d2;  // second derivative
		};

		inline Parameter Exponential(const Parameters::Parameter& f) { float ef = expf(f.value); return { f.p, ef, f.d * ef, ef * (f.d2 + f.d * f.d) }; }  // x --> exp(x)
		inline Parameter Logarithm(const Parameters::Parameter& f) { float q = f.d / f.value; return { f.p, logf(f.value), q, -q * q + f.d2 / f.value }; }  // x --> log(x)
		inline Parameter Power(float alpha, const Parameters::Parameter& f) { float pf = powf(f.value, alpha); return { f.p, pf, alpha * f.d * pf / f.value, alpha * f.d2 * pf / f.value + alpha * (alpha - 1) * pf / (f.value * f.value) }; }  // x --> x^alpha
		inline Parameter Sqrt(const Parameters::Parameter& f) { float rf = sqrtf(f.value); return { f.p, rf, f.d / (2 * rf), (4 * f.d2 * f.value - f.d * f.d) / (8 * f.value * rf) }; }  // x --> sqrt(x)
		inline Parameter Reciprocal(const Parameters::Parameter& f) { return f.reciprocal(); }  // x --> 1 / x
		inline Parameter Linear(float a, float b, const Parameters::Parameter& f) { return f * a + b; }  // x --> ax + b
		inline Parameter HypSine(const Parameters::Parameter& f) { return (Exponential(f) - Exponential(f * (-1))) / 2; }  // x --> sinh(x)
		inline Parameter HypCosine(const Parameters::Parameter& f) { return (Exponential(f) + Exponential(f * (-1))) / 2; }  // x --> cosh(x)
		inline Parameter Sine(const Parameters::Parameter& f) { float cf = cosf(f.value), sf = sinf(f.value); return { f.p, sf, cf * f.d, -sf * f.d * f.d + cf * f.d2 }; }  // x --> sin(x)
		inline Parameter Cosine(const Parameters::Parameter& f) { float cf = cosf(f.value), sf = sinf(f.value); return { f.p, cf, -sf * f.d, -cf * f.d * f.d - sf * f.d2 }; }  // x --> cos(x)
		inline Parameter Tangent(const Parameters::Parameter& f) { return Sine(f) / Cosine(f); }  // x --> tan(x)
	}

	// Should be called at the beginning of every frame
	void NewFrame();

	// Creates a bounding box in which all control points are clipped.  This has
	// its own coordinate system with x-coords and y-coords lying between -1 and
	// 1.  It cannot currently be restarted, unlike an ImGui Window.
	bool BeginView(const char* name, ImVec2 const& size, ImVec4 const& border_color, bool deferchanges = false);

	// Ends the view, applying any parameter changes, unless deferchanges is true
	void EndView();
	// Also creates a button in the view's place, which handles inputs that miss a control point.
	bool EndViewAsButton(ImGuiWindowFlags flags = 0);

	ImVec2 GetViewBoundsMin();  // In screen coords
	ImVec2 GetViewBoundsMax();  // In screen coords
	ImVec2 GetViewSize();       // In screen coords
	float GetViewAspectRatio();

	// Typically we want to apply changes at the end of the view, however if we
	// wish to bring some changes forward we can use a deferral slot, changes
	// will be applied when the slot is popped.
	void PushDeferralSlot();
	void PopDeferralSlot();

	// A composite transformation is represented by a single level on the stack,
	// new transformations are applied in place.  This has a minor performance
	// benefit and also means that the whole composite can be popped back using
	// a single call of PopTransformation.  Composite transformations can be
	// nested. No PopTransformation can occur within a composite transformation.
	void BeginCompositeTransformation();
	void EndCompositeTransformation();

	ImVec2 GetLastControlPointPosition();  // In screen coords
	float GetLastControlPointRadius();     // In screen coords

	// The regularisation parameter controls how aggressively parameters are
	// changed.  Will change this to use a stack of values.
	void SetRegularisationParameter(float kappa);

	// Applies the transformation in the stack to the given coordinate
	ImVec2 ApplyTransformation(ImVec2 const& pos);              // in view coords
	ImVec4 ApplyTransformation(ImVec4 const& pos);              // in view coords
	ImVec2 ApplyTransformationScreenCoords(ImVec2 const& pos);  // in screen coords
	ImVec2 ApplyTransformationScreenCoords(ImVec4 const& pos);  // in screen coords

	// Free parameters can be changed by a control point
	void PushFreeParameter(float* param);
	void PopFreeParameter();
	void RestrictParameter(float* param);  // Removes a given parameter from the stack, retaining the order
	void ClearFreeParameters();

	// Behaves like an invisible button at the transformed position
	bool Button(const char* str_id, ImVec2 const& pos, ImControlPointFlags flags, ImGuiButtonFlags button_flags, float marker_radius, ImVec4 marker_col, float z_order);
	bool Button(const char* str_id, ImVec4 const& pos, ImControlPointFlags flags, ImGuiButtonFlags button_flags, float marker_radius, ImVec4 marker_col);
	
	// A control point which can be clicked and dragged, changing the free parameters
	bool Point(const char* str_id, ImVec2 const& pos, ImControlPointFlags flags, ImGuiButtonFlags button_flags, float marker_radius, ImVec4 marker_col, float z_order);
	bool Point(const char* str_id, ImVec4 const& pos, ImControlPointFlags flags, ImGuiButtonFlags button_flags, float marker_radius, ImVec4 marker_col);
	
	// Transformations 
	//
	// Parameters are passed either by value, reference or by pointer. If passed
	// by value or reference they are considered **constant**, they will not be
	// considered when changing parameters.  If passed by a pointer then the
	// parameter can be changed. To avoid mistakenly passing one type instead of
	// the other, all functions taking only values or references will have
	// "Constant" in their names.  Note that some functions may have a mixture
	// of both.

	void PushConstantMatrix(ImMat4 const& M);
	void PushConstantPerspectiveMatrix(float fov, float aspect_ratio, float z_near, float z_far);  // All parameters are constants
	void PushConstantLookAtMatrix(ImVec4 const& eye, ImVec4 const& center, ImVec4 const& up);      // All parameters are constants

	void PushRotationAboutAxis(float* parameter, ImVec4 const& axis);
	void PushConstantRotationAboutAxis(float angle, ImVec4 const& axis);

	void PushRotationAboutX(float* parameter);
	void PushRotationAboutY(float* parameter);
	void PushRotationAboutZ(float* parameter);
	void PushRotationAboutX(const Parameters::Parameter& param);
	void PushRotationAboutY(const Parameters::Parameter& param);
	void PushRotationAboutZ(const Parameters::Parameter& param);
	void PushConstantRotationAboutX(float angle);
	void PushConstantRotationAboutY(float angle);
	void PushConstantRotationAboutZ(float angle);

	void PushRotation(float* parameter);  // 2d cases
	void PushRotation(const Parameters::Parameter& param);
	void PushConstantRotation(float angle);

	void PushScale(float* parameter);
	void PushScaleX(float* parameter);
	void PushScaleY(float* parameter);
	void PushScaleZ(float* parameter);
	void PushScaleAlongAxis(float* parameter, const ImVec4& axis);
	void PushScaleFromAxis(float* parameter, const ImVec4& axis);
	void PushScale(const Parameters::Parameter& param);
	void PushScaleX(const Parameters::Parameter& param);
	void PushScaleY(const Parameters::Parameter& param);
	void PushScaleZ(const Parameters::Parameter& param);
	void PushScaleAlongAxis(const Parameters::Parameter& param, const ImVec4& axis);
	void PushScaleFromAxis(const Parameters::Parameter& param, const ImVec4& axis);
	void PushConstantScale(float factor);
	void PushConstantScaleX(float factor);
	void PushConstantScaleY(float factor);
	void PushConstantScaleZ(float factor);
	void PushConstantScaleAlongAxis(float factor, const ImVec4& axis);
	void PushConstantScaleFromAxis(float factor, const ImVec4& axis);
	void PushConstantReflection(const ImVec4& normal);

	void PushTranslation(float* parameter, ImVec4 const& v);
	void PushTranslation(float* parameter, ImVec2 const& v);
	void PushTranslation(const Parameters::Parameter& param, ImVec4 const& v);
	void PushTranslation(const Parameters::Parameter& param, ImVec2 const& v);
	void PushTranslationAlongX(float* parameter);
	void PushTranslationAlongY(float* parameter);
	void PushTranslationAlongZ(float* parameter);
	void PushTranslationAlongX(const Parameters::Parameter& param);
	void PushTranslationAlongY(const Parameters::Parameter& param);
	void PushTranslationAlongZ(const Parameters::Parameter& param);
	void PushConstantTranslation(float value, ImVec4 const& v);
	void PushConstantTranslation(float value, ImVec2 const& v);
	void PushConstantTranslationAlongX(float value);
	void PushConstantTranslationAlongY(float value);
	void PushConstantTranslationAlongZ(float value);

	void PopTransformation();
	void ClearTransformations();

	// Composite transformations
	void PushRotationAboutPoint(float* angle, const ImVec2& p);
	void PushScaleXY(float* factor);
	void PushTranslation(float* x, float* y);
	void PushTranslation(float* x, float* y, float* z);

	// Note that this only retrieves the derivative if also called in the previous frame, it may return 0 otherwise
	ImVec4 GetDerivativeAt(ImVec4 const& v, float* parameter);
	ImVec4 GetSecondDerivativeAt(ImVec4 const& v, float* p1, float* p2);

	ImMat4 GetTransformationMatrix();

	int GetTransformationStackWidth();

	void ShowDemoWindow(bool*);
	void ShowTestsWindow(bool*);
	void ShowBenchmarkWindow(bool*);

}