#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdexcept>
namespace nvinfer1
{
	namespace rt
	{

		namespace detail
		{
			int clz(int x) {
				for (int i = 31; i >= 0; --i)
					if ((1 << i) & x) return 31 - i;
				return 32;
			}
			#define INPLACE_IS_POW_2(x) (0 == ((x) & ((x) - 1)))
			int find_log_2(int x, bool round_up = false) {
				int a = 31 - clz(x);
				if (round_up) a += !INPLACE_IS_POW_2(x);
				return a;
			}
			//void find_divisor(int denom, unsigned int& mul_coeff, unsigned int& shift_coeff);
			void find_divisor(int denom, unsigned int& mul_coeff, unsigned int& shift_coeff) {
				if (denom == 0) {
					throw std::invalid_argument("Trying to find reduced divisor for 0");
				}
				if (denom == 1) {
					mul_coeff = 0;
					shift_coeff = 0;
					return;
				}
				unsigned int p = 31 + find_log_2((int)denom, true);
				unsigned int m = ((1ull << p) + denom - 1) / denom;
				mul_coeff = m;
				shift_coeff = p - 32;
			}

			__host__ __device__ __forceinline__ unsigned int umulhi(unsigned int x, unsigned int y)
			{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 100
				return __umulhi(x, y);
#else
				unsigned long long z = (unsigned long long) x * (unsigned long long) y;
				return (unsigned int)(z >> 32);
#endif
			}

			/*
			 * This is a weird implementation that returns div_up(0,1)=0 but
			 * div_up(0,2)=1 (wrong) -- just do not use it with a=0.
			 */
			__host__ __device__ inline int div_up(int a, int b)
			{
				return (a - 1) / b + 1;
			}

		} // end namespace detail

		class reduced_divisor
		{
		public:
			reduced_divisor() = default;
			__host__ __forceinline__ reduced_divisor(int _y)
				: y(_y)
			{
				detail::find_divisor(y, mul_coeff, shift_coeff);
			}
			__host__ __device__ __forceinline__ reduced_divisor(unsigned _mul_coeff, unsigned _shift_coeff, int _y)
				: mul_coeff(_mul_coeff)
				, shift_coeff(_shift_coeff)
				, y(_y)
			{
			}
			__host__ __device__ __forceinline__ int div(int x) const
			{
				/*
				 * if dividing by 1, then find_divisor wouldn't have worked because
				 * mul_coeff would have had to be 2^32, which can't be represented,
				 * so we have to special case that one.
				 */
				return (y != 1) ? detail::umulhi((unsigned int)x, mul_coeff) >> shift_coeff : x;
			}
			__host__ __device__ __forceinline__ int mod(int x) const
			{
				return x - (div(x) * y);
			}
			__host__ __device__ __forceinline__ void divmod(int x, int& q, int& mod) const
			{
				q = div(x);
				mod = x - (q * y);
			}
			__host__ __device__ __forceinline__ int get() const
			{
				return y;
			}
			inline __host__ void get_mul_shift(unsigned& mul, unsigned& shift)
			{
				mul = mul_coeff;
				shift = shift_coeff;
			}

		protected:
			unsigned int mul_coeff;
			unsigned int shift_coeff;
			int y;
		};
	}
}