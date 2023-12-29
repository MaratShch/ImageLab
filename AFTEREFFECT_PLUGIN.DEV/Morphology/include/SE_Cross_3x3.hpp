#pragma once

#include "SE_Interface.hpp"

typedef std::array<SE_Type, 9> SE_CROSS_3x3;

class SE_Cross_3x3 : public SE_Interface
{
	public:
		SE_Cross_3x3() = default;
		virtual ~SE_Cross_3x3() { ; }
		const SE_Type* GetStructuredElement (size_t& line) const {line = 3; return m_se.data();}

	private:
		const SE_CROSS_3x3 m_se = { 0, 1, 0,
					                1, 1, 1,
							        0, 1, 0 };
};