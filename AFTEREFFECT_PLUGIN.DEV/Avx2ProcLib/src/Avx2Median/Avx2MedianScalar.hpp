namespace Scalar
{
	template <typename T>
	static bool scalar_median_filter_3x3_BGRA_4444_8u
	(
		T* __restrict pInImage,
		T* __restrict pOutImage,
		A_long sizeY,
		A_long sizeX,
		A_long linePitch
	) noexcept
	{
		/* input buffer to small for perform median 3x3 */
		if (sizeX < 3 || sizeY < 3)
			return false;

		return true;
	}

	template <typename T>
	static bool scalar_median_filter_3x3_BGRA_4444_8u_luma_only
	(
		T* __restrict pInImage,
		T* __restrict pOutImage,
		A_long sizeY,
		A_long sizeX,
		A_long linePitch
	) noexcept
	{
		/* input buffer to small for perform median 3x3 */
		if (sizeX < 3 || sizeY < 3)
			return false;

		return true;
	}

}; /* namespace Scalar */

