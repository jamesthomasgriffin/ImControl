#pragma once

#include "imgui_internal.h"  // Would like to remove this, possible only needed for ImRect

namespace {
    // If imgui_internal.h has not been included then define the operations from it that we use
#ifndef IMGUI_DEFINE_MATH_OPERATORS
    inline ImVec4 operator+(const ImVec4& lhs, const ImVec4& rhs) { return ImVec4(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z, lhs.w + rhs.w); }
    inline ImVec4 operator-(const ImVec4& lhs, const ImVec4& rhs) { return ImVec4(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z, lhs.w - rhs.w); }

    inline ImVec2 operator*(const ImVec2& lhs, const float rhs) { return ImVec2(lhs.x * rhs, lhs.y * rhs); }
    inline ImVec2 operator/(const ImVec2& lhs, const float rhs) { return ImVec2(lhs.x / rhs, lhs.y / rhs); }
    inline ImVec2 operator+(const ImVec2& lhs, const ImVec2& rhs) { return ImVec2(lhs.x + rhs.x, lhs.y + rhs.y); }
    inline ImVec2 operator-(const ImVec2& lhs, const ImVec2& rhs) { return ImVec2(lhs.x - rhs.x, lhs.y - rhs.y); }
    inline ImVec2 operator*(const ImVec2& lhs, const ImVec2& rhs) { return ImVec2(lhs.x * rhs.x, lhs.y * rhs.y); }
    inline ImVec2 operator/(const ImVec2& lhs, const ImVec2& rhs) { return ImVec2(lhs.x / rhs.x, lhs.y / rhs.y); }
    inline ImVec2& operator*=(ImVec2& lhs, const float rhs) { lhs.x *= rhs; lhs.y *= rhs; return lhs; }
    inline ImVec2& operator/=(ImVec2& lhs, const float rhs) { lhs.x /= rhs; lhs.y /= rhs; return lhs; }
    inline ImVec2& operator+=(ImVec2& lhs, const ImVec2& rhs) { lhs.x += rhs.x; lhs.y += rhs.y; return lhs; }
    inline ImVec2& operator-=(ImVec2& lhs, const ImVec2& rhs) { lhs.x -= rhs.x; lhs.y -= rhs.y; return lhs; }
    inline ImVec2& operator*=(ImVec2& lhs, const ImVec2& rhs) { lhs.x *= rhs.x; lhs.y *= rhs.y; return lhs; }
    inline ImVec2& operator/=(ImVec2& lhs, const ImVec2& rhs) { lhs.x /= rhs.x; lhs.y /= rhs.y; return lhs; }
#endif

    // Fill in the operators we require but that ImGui internal does not provide
    inline ImVec4 operator*(const ImVec4& lhs, const float rhs) { return ImVec4(lhs.x * rhs, lhs.y * rhs, lhs.z * rhs, lhs.w * rhs); }
    inline ImVec4 operator/(const ImVec4& lhs, const float rhs) { return ImVec4(lhs.x / rhs, lhs.y / rhs, lhs.z / rhs, lhs.w / rhs); }
    inline ImVec4& operator*=(ImVec4& lhs, const float rhs) { lhs.x *= rhs; lhs.y *= rhs; lhs.z *= rhs; lhs.w *= rhs; return lhs; }
    inline ImVec4& operator/=(ImVec4& lhs, const float rhs) { lhs.x /= rhs; lhs.y /= rhs; lhs.z /= rhs; lhs.w /= rhs; return lhs; }
    inline ImVec4& operator+=(ImVec4& lhs, const ImVec4& rhs) { lhs.x += rhs.x; lhs.y += rhs.y; lhs.z += rhs.z; lhs.w += rhs.w; return lhs; }
    inline ImVec4& operator-=(ImVec4& lhs, const ImVec4& rhs) { lhs.x -= rhs.x; lhs.y -= rhs.y; lhs.z -= rhs.z; lhs.w -= rhs.w; return lhs; }
    inline ImVec4& operator/=(ImVec4& lhs, const ImVec4& rhs) { lhs.x /= rhs.x; lhs.y /= rhs.y; lhs.z /= rhs.z; lhs.w /= rhs.w; return lhs; }

    // Some simple geometric functions
    inline float dot(ImVec4 const& a, ImVec4 const& b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }
    inline float length(ImVec4 const& v) { return sqrtf(dot(v, v)); }
    inline ImVec4 normalise(ImVec4 const& v) { return v / length(v); }
    inline ImVec4 cross(ImVec4 const& v, ImVec4 const& w) { return ImVec4{ v.y * w.z - v.z * w.y, v.z * w.x - v.x * w.z, v.x * w.y - v.y * w.x, 0.0f }; }

    // Operators for matrix type - not complete, just the ones we need
    inline ImVec4 operator*(ImMat4 const& A, ImVec4 const& v) { return A.col[0] * v.x + A.col[1] * v.y + A.col[2] * v.z + A.col[3] * v.w; }
    inline ImMat4 operator*(ImMat4 const& A, ImMat4 const& B) { return ImMat4{ A * B.col[0], A * B.col[1], A * B.col[2], A * B.col[3] }; }
    inline ImMat4 operator*(float a, ImMat4 const& B) { return ImMat4{ B.col[0] * a, B.col[1] * a, B.col[2] * a, B.col[3] * a }; }
    inline ImMat4 operator+(ImMat4 const& A, ImMat4 const& B) { return ImMat4{ A.col[0] + B.col[0], A.col[1] + B.col[1], A.col[2] + B.col[2], A.col[3] + B.col[3] }; }
    inline ImMat4& operator+=(ImMat4& A, ImMat4 const& B) { A.col[0] += B.col[0]; A.col[1] += B.col[1]; A.col[2] += B.col[2]; A.col[3] += B.col[3]; return A; }
    // NB *= is right multiplication
    inline ImMat4& operator*=(ImMat4& A, ImMat4 const& B) { A = A * B; return A; }
}  // end anonymous namespace

namespace ImControl {
    // Parameters are recognised by a pointer to their values
    using param_t = float*;

    namespace Transformations
    {
        // The classes below implement parametrised transformations, precisely this
        // is a (twice differentiable) function T(t) from real numbers to 4x4
        // matrices.  Here the matrices act on vectors by left multiplication, i.e.
        // v ---> T(t)v.  Such a function can be differentiated with respect to t to
        // get a matrix T'(t) and then again to get a matrix T''(t).
        //
        // The classes should be initialised with the required value of t (and any
        // other required parameters). The only functionally the classes need
        // implement is right multiplication by each of the matrices T(t), T'(t) and
        // T''(t).  NB this is not the usual left action on vectors. The member
        // functions has the signatures,
        //
        // ImMat4& ApplyTransformationOnRightInPlace(ImMat4*, ImMat4*) const,
        // ImMat4& ApplyDerivativeOnRightInPlace(ImMat4&) const, ImMat4&
        // Apply2ndDerivOnRightInPlace(ImMat4&) const.
        //
        // The first calculates the action M ---> M T(t), over an array of matrices
        // M specified by begin and end pointers.  The second calculates M ---> M
        // T'(t) in place for the matrix M passed by reference.  The third is the
        // same but for M ---> M T''(t).
        //
        // These implementation choices were taken with efficiency in mind. When
        // implementing one should keep in mind that it is the base transformation
        // this is applied most often in the object's lifetime and one should assume
        // that it could be applied 1-20 times, while the first derivative might be
        // applied 0-5 times and the second derivative will only be applied at most
        // once and usually not at all.  So for complicated transformations it might
        // be worth caching the transformation matrix, may be counterproductive
        // caching the first derivative (if it is not used) and one should never
        // cache the second derivative matrix.

        using scalar_t = float;  // Perhaps it should be an option to use double precision

        template<unsigned int C>
        class TranslationOfCoordinate {
            const scalar_t m_t;
        public:
            TranslationOfCoordinate(scalar_t t) : m_t{ t } {};
            inline void applyTransformationOnRightInPlace(ImMat4* m_begin, ImMat4* m_end) const {
                for (ImVec4* it = m_begin->col; it < m_end->col; it += 4)
                    it[3] += it[C] * m_t;
            }
            inline ImMat4& applyDerivativeOnRightInPlace(ImMat4& M) const {
                M.col[3] = M[C];
                M.col[0] = M.col[1] = M.col[2] = {};
                return M;
            }
            inline ImMat4& apply2ndDerivOnRightInPlace(ImMat4& M) const {
                M = {};
                return M;
            }
        };

        class Translation {
            const scalar_t m_t;
            const ImVec4 m_dir;
        public:
            Translation(const ImVec4& dir, scalar_t t) : m_t{ t }, m_dir{ dir } { IM_ASSERT(m_dir.w == 0.0f); };
            inline void applyTransformationOnRightInPlace(ImMat4* m_begin, ImMat4* m_end) const {
                for (ImVec4* it = m_begin->col; it < m_end->col; it += 4)
                    it[3] += it[0] * (m_dir.x * m_t) + it[1] * (m_dir.y * m_t) + it[2] * (m_dir.z * m_t);
            }
            inline ImMat4& applyDerivativeOnRightInPlace(ImMat4& M) const {
                M.col[3] = M.col[0] * m_dir.x + M.col[1] * m_dir.y + M.col[2] * m_dir.z;
                M.col[0] = M.col[1] = M.col[2] = {};
                return M;
            }
            inline ImMat4& apply2ndDerivOnRightInPlace(ImMat4& M) const {
                M = {};
                return M;
            }
        };

        class ConstantMatrix {
            const ImMat4 m_M;
        public:
            ConstantMatrix(const ImMat4& M) : m_M{ M } {};
            inline void applyTransformationOnRightInPlace(ImMat4* m_begin, ImMat4* m_end) const {
                for (ImMat4* it = m_begin; it != m_end; ++it)
                    *it = *it * m_M;
            }
            inline ImMat4& applyDerivativeOnRightInPlace(ImMat4& M) const {
                M = {};
                return M;
            }
            inline ImMat4& apply2ndDerivOnRightInPlace(ImMat4& M) const {
                M = {};
                return M;
            }
        };

        template<unsigned int C>
        class ScaleCoordinate {
            const scalar_t m_t;
        public:
            ScaleCoordinate(scalar_t t) : m_t{ t } {};
            inline void applyTransformationOnRightInPlace(ImMat4* m_begin, ImMat4* m_end) const {
                for (ImMat4* it = m_begin; it < m_end; ++it)
                    it->operator[](C) *= m_t;
            }
            inline ImMat4& applyDerivativeOnRightInPlace(ImMat4& M) const {
                ImVec4 m = M[C];
                M = {};
                M[C] = m;
                return M;
            }
            inline ImMat4& apply2ndDerivOnRightInPlace(ImMat4& M) const {
                M = {};
                return M;
            }
        };

        class Scale {
            const scalar_t m_t;
        public:
            Scale(scalar_t t) : m_t{ t } {};
            inline void applyTransformationOnRightInPlace(ImMat4* m_begin, ImMat4* m_end) const {
                for (ImMat4* it = m_begin; it < m_end; ++it) {
                    it->operator[](0) *= m_t;
                    it->operator[](1) *= m_t;
                    it->operator[](2) *= m_t;
                }
            }
            inline ImMat4& applyDerivativeOnRightInPlace(ImMat4& M) const {
                M[3] = {};
                return M;
            }
            inline ImMat4& apply2ndDerivOnRightInPlace(ImMat4& M) const {
                M = {};
                return M;
            }
        };

        class ScaleAboutAxis {
            const scalar_t m_t;
            const ImVec4 m_dir;
        public:
            ScaleAboutAxis(scalar_t t, ImVec4 const& dir) : m_t{ t }, m_dir{ normalise(dir) } { IM_ASSERT(dir.w == 0.0f); };
            inline void applyTransformationOnRightInPlace(ImMat4* m_begin, ImMat4* m_end) const {
                for (ImMat4* it = m_begin; it < m_end; ++it) {
                    ImVec4 a = ((*it) * m_dir) * (1 - m_t);
                    it->operator[](0) = it->operator[](0) * m_t + a * m_dir.x;
                    it->operator[](1) = it->operator[](1) * m_t + a * m_dir.y;
                    it->operator[](2) = it->operator[](2) * m_t + a * m_dir.z;
                }
            }
            inline ImMat4& applyDerivativeOnRightInPlace(ImMat4& M) const {
                ImVec4 a = M * m_dir * -1;
                M[0] += a * m_dir.x;
                M[1] += a * m_dir.y;
                M[2] += a * m_dir.z;
                M[3] = {};
                return M;
            }
            inline ImMat4& apply2ndDerivOnRightInPlace(ImMat4& M) const {
                M = {};
                return M;
            }
        };

        class ScaleInDirection {
            const scalar_t m_t;
            const ImVec4 m_dir;
        public:
            ScaleInDirection(scalar_t t, ImVec4 const& dir) : m_t{ t }, m_dir{ normalise(dir) } { IM_ASSERT(dir.w == 0.0f); };
            inline void applyTransformationOnRightInPlace(ImMat4* m_begin, ImMat4* m_end) const {
                for (ImMat4* it = m_begin; it < m_end; ++it) {
                    ImVec4 a = ((*it) * m_dir) * (m_t - 1);
                    it->operator[](0) += a * m_dir.x;
                    it->operator[](1) += a * m_dir.y;
                    it->operator[](2) += a * m_dir.z;
                }
            }
            inline ImMat4& applyDerivativeOnRightInPlace(ImMat4& M) const {
                ImVec4 a = M * m_dir;
                M[0] = a * m_dir.x;
                M[1] = a * m_dir.y;
                M[2] = a * m_dir.z;
                M[3] = {};
                return M;
            }
            inline ImMat4& apply2ndDerivOnRightInPlace(ImMat4& M) const {
                M = {};
                return M;
            }
        };

        // Is specialised to rotations about a particular axis
        template<unsigned int I, unsigned int J>
        class RotationIJ {
            const scalar_t m_c, m_s;  // cache of cos(t) and sin(t)
        public:
            RotationIJ(scalar_t t) : m_c{ cosf(t) }, m_s{ sinf(t) } {}
            inline void applyTransformationOnRightInPlace(ImMat4* m_begin, ImMat4* m_end) const {
                ImVec4* const end_col = m_end->col;
                ImVec4* itI = m_begin->col + I;
                ImVec4* itJ = m_begin->col + J;
                while (itI < end_col) {
                    ImVec4 temp = *itI;  // copy
                    *itI = *itI * m_c + *itJ * m_s;
                    *itJ = temp * (-m_s) + *itJ * m_c;
                    itI += 4; itJ += 4;
                }
            }
            inline ImMat4& applyDerivativeOnRightInPlace(ImMat4& M) const {
                ImVec4 mI = M[I];
                ImVec4 mJ = M[J];
                M = {};  // Zero M
                M[I] = mI * (-m_s) + mJ * m_c;
                M[J] = mI * (-m_c) - mJ * m_s;
                return M;
            }
            inline ImMat4& apply2ndDerivOnRightInPlace(ImMat4& M) const {
                ImVec4 mI = M[I];
                ImVec4 mJ = M[J];
                M = {};  // Zero M
                M[I] = mI * (-m_c) - mJ * m_s;
                M[J] = mI * m_s - mJ * m_c;
                return M;
            }
        };

        using RotationX = RotationIJ<1, 2>;
        using RotationY = RotationIJ<2, 0>;
        using RotationZ = RotationIJ<0, 1>;

        class RotationAroundAxis {
            ImMat4 m_M{};  // cache of rotation matrix
            ImVec4 m_axis{};
            const scalar_t m_c, m_s;  // cache of cos(t) and sin(t)
        public:
            RotationAroundAxis(const ImVec4& axis, scalar_t t) : m_axis{ normalise(axis) }, m_c{ cosf(t) }, m_s{ sinf(t) } {
                IM_ASSERT(axis.w == 0.0f);  // We could allow non-zero values but then should project the axis in the proper way

                // Calculate rotation matrix in constructor, adapted from the glm implementation of rotate
                ImVec4 temp = m_axis * (1 - m_c);
                m_M[0].x = m_c + temp.x * m_axis.x;
                m_M[0].y = temp.x * m_axis.y + m_s * m_axis.z;
                m_M[0].z = temp.x * m_axis.z - m_s * m_axis.y;

                m_M[1].x = temp.y * m_axis.x - m_s * m_axis.z;
                m_M[1].y = m_c + temp.y * m_axis.y;
                m_M[1].z = temp.y * m_axis.z + m_s * m_axis.x;

                m_M[2].x = temp.z * m_axis.x + m_s * m_axis.y;
                m_M[2].y = temp.z * m_axis.y - m_s * m_axis.x;
                m_M[2].z = m_c + temp.z * m_axis.z;

                m_M[3].w = 1.0f;
            };
            inline void applyTransformationOnRightInPlace(ImMat4* m_begin, ImMat4* m_end) const {
                for (ImMat4* it = m_begin; it != m_end; ++it)
                    *it = *it * m_M;
            }
            inline ImMat4& applyDerivativeOnRightInPlace(ImMat4& M) const {
                ImVec4 temp = m_axis * m_s;

                ImMat4 dM{};
                dM[0].x = -m_s + temp.x * m_axis.x;
                dM[0].y = temp.x * m_axis.y + m_c * m_axis.z;
                dM[0].z = temp.x * m_axis.z - m_c * m_axis.y;

                dM[1].x = temp.y * m_axis.x - m_c * m_axis.z;
                dM[1].y = -m_s + temp.y * m_axis.y;
                dM[1].z = temp.y * m_axis.z + m_c * m_axis.x;

                dM[2].x = temp.z * m_axis.x + m_c * m_axis.y;
                dM[2].y = temp.z * m_axis.y - m_c * m_axis.x;
                dM[2].z = -m_s + temp.z * m_axis.z;

                M = M * dM;
                return M;
            }
            inline ImMat4& apply2ndDerivOnRightInPlace(ImMat4& M) const {
                ImVec4 temp = m_axis * m_c;

                ImMat4 d2M{};
                d2M[0].x = -m_c + temp.x * m_axis.x;
                d2M[0].y = temp.x * m_axis.y - m_s * m_axis.z;
                d2M[0].z = temp.x * m_axis.z + m_s * m_axis.y;

                d2M[1].x = temp.y * m_axis.x + m_s * m_axis.z;
                d2M[1].y = -m_c + temp.y * m_axis.y;
                d2M[1].z = temp.y * m_axis.z - m_s * m_axis.x;

                d2M[2].x = temp.z * m_axis.x - m_s * m_axis.y;
                d2M[2].y = temp.z * m_axis.y + m_s * m_axis.x;
                d2M[2].z = -m_c + temp.z * m_axis.z;

                M = M * d2M;
                return M;
            }
        };


        // The transformation stack holds matrix representations of transformations and their (second) derivatives
        class TransformationStack
        {
        public:

            inline ImVec4 apply(const ImVec4& p) const { return isEmpty() ? p : getMatrix() * p; };
            inline ImVec4 applyDerivative(const ImVec4& v, float* param) const { return isEmpty() ? ImVec4{} : getDerivativeMatrix(param) * v; };
            inline ImVec4 applySecondDerivative(const ImVec4& v, float* p1, float* p2) const { return isEmpty() ? ImVec4{} : getSecondDerivativeMatrix(p1, p2) * v; };


            inline const ImMat4& getMatrix() const { return isEmpty() ? m_identity : *beginLevel(); }
            const ImMat4& getDerivativeMatrix(param_t param) const;
            const ImMat4& getSecondDerivativeMatrix(param_t p1, param_t p2) const;

            inline size_t getStackCapacity() const { return static_cast<size_t>(m_data.capacity()); }  // In number of matrices
            inline size_t getStackDepth() const { return m_depth; }  // In number of transformations
            inline size_t getStackSize() const { return m_data.Size; }  // In number of matrices
            inline size_t getStackWidth() const { return m_width; }  // In number of matrices

            inline size_t getNumDerivatives() const { return static_cast<size_t>(m_derivative_parameters.size()); }
            inline size_t getNumSecondDerivatives() const { return m_num_second_derivs; }


            void setDerivativeParameters(const ImVector<float*>& d_params, int n_second_derivatives);

            void pushCompositeLevel() { if (m_composite_level == 0) pushIdentity(); ++m_composite_level; }
            void popCompositeLevel() { IM_ASSERT(m_composite_level > 0); m_composite_level--; };

            template<typename T_t> void pushConstantTransformation(const T_t& T);
            template<typename T_t> void pushTransformation(const T_t& T, param_t param);
            template<typename T_t> void pushTransformation(const T_t& T, const Parameters::Parameter& param);

            inline void popTransformation() { IM_ASSERT(!isEmpty()); IM_ASSERT(m_composite_level == 0); resize(m_depth - 1); }

            inline bool isEmpty() const { return m_depth == 0; }
            void clear() { resize(0); };
            void reset();
            inline void resize(size_t new_size) { m_data.resize(static_cast<int>(new_size * m_width)); m_depth = new_size; }


        private:
            // This is here so that we can return references to the identity when the stack is empty (could be made static)
            const ImMat4 m_identity{ ImVec4{1,0,0,0}, ImVec4{0,1,0,0}, ImVec4{0,0,1,0}, ImVec4{0,0,0,1} };

            // This is here so that we can return references to zero when a parameter isn't in the list of derivatives (could be made static)
            const ImMat4 m_zero_matrix{ ImVec4{0,0,0,0}, ImVec4{0,0,0,0}, ImVec4{0,0,0,0}, ImVec4{0,0,0,0} };

            size_t m_depth{};  // depth of stack
            size_t m_width{ 1 };  // width of stack, determined by number of derivatives required

            ImVector<ImMat4> m_data{};  // should have size m_depth * m_width at any time

            // Contains the list of parameters we are differentiating against
            ImVector<param_t> m_derivative_parameters{};

            // We compute second derivatives for all pairs of parameters in the first m_num_second_derivs entries of m_derivative_parameters
            size_t m_num_second_derivs{};

            // The depth of composite transformation calls
            unsigned int m_composite_level{};

            void copyWithinStack(size_t source_ix, size_t target_ix);
            void pushIdentity();

            inline ImMat4* beginLevel() { return m_data.Data + (m_depth - 1) * m_width; }
            inline ImMat4* endLevel() { return m_data.Data + m_depth * m_width; }
            inline ImMat4 const* beginLevel() const { return m_data.Data + (m_depth - 1) * m_width; }
            inline ImMat4 const* endLevel() const { return m_data.Data + m_depth * m_width; }

            inline ImMat4* beginDerivatives() { return m_data.Data + (m_depth - 1) * m_width + 1; }
            inline ImMat4* endDerivatives() { return m_data.Data + (m_depth - 1) * m_width + 1 + static_cast<size_t>(m_derivative_parameters.Size); }
            inline ImMat4 const* beginDerivatives() const { return m_data.Data + (m_depth - 1) * m_width + 1; }
            inline ImMat4 const* endDerivatives() const { return m_data.Data + (m_depth - 1) * m_width + 1 + static_cast<size_t>(m_derivative_parameters.Size); }

            inline ImMat4* beginSecondDerivatives() { return m_data.Data + (m_depth - 1) * m_width + 1 + static_cast<size_t>(m_derivative_parameters.Size); }
            inline ImMat4* endSecondDerivatives() { return endLevel(); }
            inline ImMat4 const* beginSecondDerivatives() const { return m_data.Data + (m_depth - 1) * m_width + 1 + static_cast<size_t>(m_derivative_parameters.Size); }
            inline ImMat4 const* endSecondDerivatives() const { return endLevel(); }
        };

        template<typename T>
        inline T symmetric_index_ordered(T i, T j) { IM_ASSERT(i <= j); return i + (j * (j + 1)) / 2; }
        template<typename T>
        inline T symmetric_index(T i, T j) { if (i > j) { /* swap */ T k = i; i = j; j = k; } return symmetric_index_ordered(i, j); }
        template<typename T> inline T symmetric_size(T n) { return symmetric_index_ordered<T>(0, n); }

        // A class that holds a list of parameters, a value of type T and the derivatives of that value with respect to the parameters
        template<typename T>
        class ImJet1 {
        public:
            const ImVector<param_t> parameters;  // The number of parameters, a constant
            ImJet1(const ImVector<param_t>& params) : parameters{ params } { int s = size(); m_data.resize(1 + s, {}); }

            int size() const { return parameters.Size; }

            T& value() { return m_data[0]; }
            const T& value() const { return m_data[0]; }
            T& derivative(int i) { IM_ASSERT(i < size()); return m_data[1 + i]; }
            const T& derivative(int i) const { IM_ASSERT(i < size()); return m_data[1 + i]; }

        private:
            ImVector<T> m_data{};
        };

        // A class that holds a list of parameters, a value of type T, the derivatives of that value with respect to the parameters and the second derivatives of that value wrt those same parameters
        template<typename T>
        class ImJet2 {
        public:
            const ImVector<param_t> parameters;  // The list of parameters, a constant
            ImJet2(const ImVector<param_t>& params) : parameters{ params } { int s = size(); m_data.resize(1 + s + symmetric_size(s), {}); }

            int size() const { return parameters.Size; }

            T& value() { return m_data[0]; }
            const T& value() const { return m_data[0]; }
            T& derivative(int i) { IM_ASSERT(i < size()); return m_data[1 + i]; }
            const T& derivative(int i) const { IM_ASSERT(i < size()); return m_data[1 + i]; }
            const T& second_derivative(int i, int j) const { IM_ASSERT((i < size()) && (j < size())); return m_data[1 + size() + symmetric_index(i, j)]; }
            T& second_derivative(int i, int j) { IM_ASSERT((i < size()) && (j < size())); return m_data[1 + size() + symmetric_index(i, j)]; }

            void push_back_parameter(param_t p) {
                parameters.push_back(p);
                auto n = parameters.size();
                m_data.insert(m_data.begin() + n, {});  // Add slot for new derivative, shifting current second derivatives back
                m_data.resize(1 + n + symmetric_size(0, n), {});  // Make space at end for new second derivatives
            }

            ImJet2& operator+=(const ImJet2& b) {
                value() += b.value();
                ImVector<int> b_param_indices{};
                for (int i = 0; i < b.size(); ++i) {
                    const param_t* q = parameters.find(b.parameters[i]);
                    if (q == parameters.end())
                        push_back_parameter(b.parameters[i]);
                    const int ix = q - parameters.begin();
                    b_param_indices.push_back(ix);
                    derivative(ix) += b.derivative(i);
                    for (int j = 0; j <= i; ++j)
                        second_derivative(b_param_indices[j], ix) += b.second_derivative(j, i);
                }
            }

        private:
            ImVector<T> m_data{};  // 1 value, n derivatives and n * (n + 1) / 2 second derivatives
        };


        ImJet1<ImVec4> apply_stack_to_1jet(const TransformationStack& transform, const ImJet1<ImVec4>& jet);
        ImJet2<ImVec4> apply_stack_to_2jet(const TransformationStack& transform, const ImJet2<ImVec4>& jet);

        ImJet1<ImVec4> project_1jet(const ImJet1<ImVec4>& jet);
        ImJet2<ImVec4> project_2jet(const ImJet2<ImVec4>& jet);

        ImJet2<float> distance_squared(const ImJet2<ImVec4>& jet, const ImVec4& target);

        ImVector<float> apply_conjugate_gradient_method(const ImJet2<float>& jet, float kappa = 1.0f);

        inline ImVec4 calculate_tangent_after_projection(const TransformationStack& transform, const ImVec4& pos, param_t param);

        ImVector<float> bring_together(const TransformationStack& transform, const ImVec4& pos, const ImVec4& target, const ImVector<param_t>& free_parameters, float kappa = 1.0f);

    }  // namespace Transformations

    struct parameter_change_t {
        param_t parameter{};
        float change{ 0 };
    };

    // The deferral stack gives control over when a parameter change happens, rather
    // than applying a change immediately the change is placed in a slot and the
    // change happens when the slot is popped from the stack.  In practice we only
    // have one change queued and so instead of having a stack of changes we just
    // record which slot the single change is contained in.
    class ParameterChangeDeferralStack {
    public:
        void addParameterChange(ImVector<parameter_change_t> const& changes);
        bool applyParameterChanges();

        void pushDeferralSlot();
        bool popDeferralSlot();

        void reset();

    private:
        ImVector<parameter_change_t> m_deferred_changes{};  // The deferred change
        unsigned int m_deferral_stack_size{ 1 };  // Stack is never empty, there is always a slot
        int m_deferred_change_position{ -1 };  // -1 means there is no deferred step in the stack
    };

    /*
    A single manager handles all the control points, this works because only one
    item is typically active at once. A number of control points may be
    activated at the same time if they overlap, but only the one with the least
    z value will be active in the next frame.

    In order to move a control point, the (second) derivatives with respect to
    its free parameters are required. The (second) derivatives are calculated as
    the stack is built, so the required parameters are registered when an item
    is activated, then the derivatives for the registered parameters are
    calculated for the next frame when the item is active.

    The user has some control of when parameters are changed.  Clearly they can
    only be changed when an item is active.  The default is that the change
    happens at the end of frame. By passing the relevant ControlPointFlag the
    user can specify that the change happens after the control point function is
    called.  Or the user can use the PushDeferralSlot() and PopDeferralSlot()
    functions. This causes the parameter to be changed when the corresponding
    slot is popped.
    */
    class ImControlContext {
    public:
        void newFrame(Transformations::TransformationStack& transform);
        void endFrame();

        bool beginView(const char* name, ImVec2 size, ImVec4 const& border_color, bool defer_changes);
        bool endView(bool include_button = false, ImGuiButtonFlags flags = 0);

        void pushDeferralSlot();
        bool popDeferralSlot();  // Returns true if any parameters are changed

        ImVec2 getViewBoundsMin() const;
        ImVec2 getViewBoundsMax() const;
        const ImVec2& getViewSize() const;
        ImVec2 const& getLastControlPointPosition() const { return m_last_control_point_position; }
        float getLastControlPointRadius() const { return m_last_control_point_radius; }
        float const& getLastParameterStep(int ix) const { return m_last_step[ix].change; }

        void registerFreeParameterDerivativeForNextFrame(param_t param);
        void registerFreeParameterSecondDerForNextFrame(param_t p);

        ImVec2 pointInScreenCoords(const Transformations::TransformationStack& transform, const ImVec4& pos);

        void setRegularisationParam(float p) { m_regularisation_constant = p; }

        // Manage vector of free parameters
        void pushFreeParameter(param_t param) { IM_ASSERT(m_free_parameters.find(param) == m_free_parameters.end()); m_free_parameters.push_back(param); }
        void popFreeParameter() { IM_ASSERT(m_free_parameters.size() > 0); m_free_parameters.pop_back(); }
        void restrictParameter(param_t param) { bool found = m_free_parameters.find_erase(param); IM_ASSERT(found); }  // Maintains order of parameters
        void clearParameters() { m_free_parameters.resize(0); }  // Maintains order of parameters

        bool staticControlPoint(const char* str_id, const Transformations::TransformationStack& transform, const ImVec4& pos, ImControlPointFlags flags = 0, ImGuiButtonFlags button_flags = 0, float marker_radius = 0.1f, ImVec4 marker_col = { 1, 1, 1, 1 });
        bool controlPoint(const char* str_id, const Transformations::TransformationStack& transform, const ImVec4& pos, ImControlPointFlags flags, ImGuiButtonFlags button_flags, float marker_radius, ImVec4 marker_col);

    private:
        bool m_view_active{ false };
        ImGuiID m_view_id{};
        bool m_view_defers_changes{ false };
        float m_view_marker_size_factor{ 1 };
        ImVec2 m_view_size{};
        ImRect m_view_bb{};

        ImVec2 m_cursor_pos_before_view{};
        ImVec2 m_cursor_position_after_view{};

        ImVector<param_t> m_free_parameters{};

        ImVector<param_t> m_registered_der_params{};
        ImVector<param_t> m_registered_2nd_der_params{};

        void drawDerivative(ImVec2 const& pos, ImVec2 const& d, ImVec4 const& col, float thickness = 1.0f) const;

        ImVec2 getMousePositionInViewCoords() const;
        ImVec2 viewCoordsToScreenCoords(ImVec2 const& p) const;
        ImVec2 screenCoordsToViewCoords(ImVec2 const& p) const;
        bool createPoint(const char* str_id, ImVec4 const& transformed_pos, ImControlPointFlags const& flags, ImGuiButtonFlags const& button_flags, float marker_radius, ImVec4 marker_col);

        ImGuiID m_activated_point_id{};
        void* m_activated_point_window{};
        float m_activated_point_z_order{};
        ImVector<param_t> m_activated_point_parameters{};

        float m_last_point_z_order{};

        bool saveActivatedPoint(const ImVector<float*>& params = {});

        ImVector<parameter_change_t> m_last_step{};
        ImVec2 m_last_control_point_position{};
        float m_last_control_point_radius{};

        ParameterChangeDeferralStack m_deferral_stack{};

        float m_regularisation_constant = 0.5f;

        bool updateParameters(ImVector<parameter_change_t> const& changes, ImControlPointFlags const& flags);
    };
}