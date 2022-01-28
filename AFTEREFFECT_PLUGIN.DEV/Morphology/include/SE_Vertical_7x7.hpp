#pragma once

#include "SE_Interface.hpp"

typedef std::array<SE_Type, 49> SE_VERTICAL_7x7;

class SE_Vertical_7x7 : public SE_Interface
{
	public:
		SE_Vertical_7x7() = default;
		virtual ~SE_Vertical_7x7() { ; }
		const SE_Type* GetStructuredElement (size_t& line) const {line = 7; return m_se.data();}

	private:
		const SE_VERTICAL_7x7 m_se = { 0, 0, 1, 1, 1, 0, 0,  
					                   0, 0, 1, 1, 1, 0, 0,
			                           0, 0, 1, 1, 1, 0, 0,
			                           0, 0, 1, 1, 1, 0, 0,
			                           0, 0, 1, 1, 1, 0, 0,
			                           0, 0, 1, 1, 1, 0, 0,
			                           0, 0, 1, 1, 1, 0, 0 };
};