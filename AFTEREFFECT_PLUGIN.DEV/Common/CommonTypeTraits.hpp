#pragma once
#include <type_traits>

template <typename T>
class is_float_32
{
	/**
	* YUV proc variation
	*/
	template <typename TT,
		typename std::enable_if<
		std::is_same<TT, float>::value>::type* = nullptr>
		static auto test(int)->std::true_type;

	template<typename>
	static auto test(...)->std::false_type;

public:
	static constexpr const bool value = decltype(test<T>(0))::value;
};

template <typename T>
class is_float_64
{
	/**
	* YUV proc variation
	*/
	template <typename TT,
		typename std::enable_if<
		std::is_same<TT, double>::value ||
		std::is_same<TT, long double>::value>::type* = nullptr>
		static auto test(int)->std::true_type;

	template<typename>
	static auto test(...)->std::false_type;

public:
	static constexpr const bool value = decltype(test<T>(0))::value;
};