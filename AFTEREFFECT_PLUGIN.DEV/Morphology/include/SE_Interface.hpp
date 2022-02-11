#pragma once

#include <cstdint>
#include <array>
#include <atomic>
#include "MorphologyEnums.hpp"

typedef	int32_t	SE_Type;

class SE_Interface {
public:
	virtual ~SE_Interface() = default;
	virtual const SE_Type* GetStructuredElement(size_t& line) const = 0;
};


SE_Interface* CreateSeInterface(const SeType& seType, const SeSize& seSize);
void DeleteSeInterface(SE_Interface* seInterafce);

constexpr std::uint64_t INVALID_INTERFACE = 0xFFFFFFFFFFFFFFFFu;