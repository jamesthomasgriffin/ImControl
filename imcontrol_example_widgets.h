#include "imcontrol.h"

namespace CustomWidgets
{
    void TranslationWidget(ImMat4* M, float axes_size, ImControlPointFlags control_point_flags, bool include_planes);
    void RotationWidget(ImMat4* M, float axes_size, ImControlPointFlags control_point_flags);
}