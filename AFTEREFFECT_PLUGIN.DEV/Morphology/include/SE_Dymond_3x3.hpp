#pragma once

#include "SE_Interface.hpp"

typedef std::array<SE_Type, 9> SE_DYMOND_3x3;

class SE_Dymond_3x3 : public SE_Interface
{
	public:
		SE_Dymond_3x3() = default;
		virtual ~SE_Dymond_3x3() { ; }
		const SE_Type* GetStructuredElement (size_t& line) const {line = 3; return m_se.data();}

	private:
		const SE_DYMOND_3x3 m_se = { 1, 1, 1,
					                 1, 1, 1,
							         0, 1, 0 };
};