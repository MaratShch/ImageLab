#pragma once

#include "SE_Interface.hpp"

typedef std::array<SE_Type, 49> SE_DYMOND_7x7;

class SE_Dymond_7x7 : public SE_Interface
{
	public:
		SE_Dymond_7x7() = default;
		virtual ~SE_Dymond_7x7() { ; }
		const SE_Type* GetStructuredElement (size_t& line) const {line = 7; return m_se.data();}

	private:
		const SE_DYMOND_7x7 m_se = { 0, 0, 1, 1, 1, 0, 0,
					                 0, 1, 1, 1, 1, 1, 0,
			                         1, 1, 1, 1, 1, 1, 1,
			                         1, 1, 1, 1, 1, 1, 1,
			                         0, 1, 1, 1, 1, 1, 0,
			                         0, 0, 1, 1, 1, 0, 0,
			                         0, 0, 0, 1, 0, 0, 0 };
};