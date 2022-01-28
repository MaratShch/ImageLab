#pragma once

#include "SE_Interface.hpp"

typedef std::array<SE_Type, 9> SE_SQUARE_3x3;

class SE_Square_3x3 : public SE_Interface
{
	public:
		SE_Square_3x3() = default;
		virtual ~SE_Square_3x3() { ; }
		const SE_Type* GetStructuredElement(size_t& line) const {line = 3; return m_se.data();}

	private:
		const SE_SQUARE_3x3 m_se = { 1, 1, 1,
					                 1, 1, 1,
							         1, 1, 1 };
};