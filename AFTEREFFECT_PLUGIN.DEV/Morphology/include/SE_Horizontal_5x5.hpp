#pragma once

#include "SE_Interface.hpp"

typedef std::array<SE_Type, 25> SE_HORIZONTAL_5x5;

class SE_Horizontal_5x5 : public SE_Interface
{
	public:
		SE_Horizontal_5x5() = default;
		virtual ~SE_Horizontal_5x5() { ; }
		const SE_Type* GetStructuredElement (size_t& line) const {line = 5; return m_se.data();}

	private:
		const SE_HORIZONTAL_5x5 m_se = { 0, 0, 0, 0, 0,
					                   0, 0, 0, 0, 0,
			                           1, 1, 1, 1, 1,
			                           0, 0, 0, 0, 0,
			                           0, 0, 0, 0, 0 };
};