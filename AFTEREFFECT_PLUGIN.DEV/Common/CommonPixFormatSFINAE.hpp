#pragma once
#include <type_traits>

template <typename T>
class is_RGB_proc
{
	/**
	* RGB proc variation
	*/
	template <typename TT,
		typename std::enable_if<
		std::is_same<TT, PF_Pixel_BGRA_8u>::value  ||
		std::is_same<TT, PF_Pixel_BGRA_16u>::value || 
		std::is_same<TT, PF_Pixel_BGRA_32f>::value ||
		std::is_same<TT, PF_Pixel_ARGB_8u>::value  ||
		std::is_same<TT, PF_Pixel_ARGB_16u>::value ||
		std::is_same<TT, PF_Pixel_ARGB_32f>::value>::type* = nullptr>
		static auto test(int)->std::true_type;

	template<typename>
	static auto test(...)->std::false_type;

public:
	static constexpr const bool value = decltype(test<T>(0))::value;
};

template <typename T>
class is_YUV_proc
{
	/**
	* YUV proc variation
	*/
	template <typename TT,
		typename std::enable_if<
		std::is_same<TT, PF_Pixel_VUYA_8u>::value  ||
		std::is_same<TT, PF_Pixel_VUYA_16u>::value ||
		std::is_same<TT, PF_Pixel_VUYA_32f>::value>::type* = nullptr>
		static auto test(int)->std::true_type;

	template<typename>
	static auto test(...)->std::false_type;

public:
	static constexpr const bool value = decltype(test<T>(0))::value;
};


template <typename T>
class is_no_alpha_channel
{
	/**
	* YUV proc variation
	*/
	template <typename TT,
		typename std::enable_if<
		std::is_same<TT, PF_Pixel_RGB_10u>::value>::type* = nullptr>
		static auto test(int)->std::true_type;

	template<typename>
	static auto test(...)->std::false_type;

public:
	static constexpr const bool value = decltype(test<T>(0))::value;
};

template <typename T>
class is_8bits_RGB_pixel
{
	/**
	* RGB 8 bits pixel width variation
	*/
	template <typename TT,
		typename std::enable_if<
		std::is_same<TT, PF_Pixel_BGRA_8u>::value ||
		std::is_same<TT, PF_Pixel_ARGB_8u>::value>::type* = nullptr>
		static auto test(int)->std::true_type;

	template<typename>
	static auto test(...)->std::false_type;

public:
	static constexpr const bool value = decltype(test<T>(0))::value;
};

// Helpers
template<typename T>
struct is_RGB_or_YUV : std::integral_constant<bool, is_RGB_proc<T>::value || is_YUV_proc<T>::value> {};

template<typename T>
struct is_RGB_Variants : std::integral_constant<bool, is_RGB_proc<T>::value || is_no_alpha_channel<T>::value> {};

template<typename T>
struct is_SupportedImageBuffer : std::integral_constant<bool, is_RGB_Variants<T>::value || is_YUV_proc<T>::value> {};