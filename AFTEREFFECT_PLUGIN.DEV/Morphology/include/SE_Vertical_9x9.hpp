#pragma once

#include "SE_Interface.hpp"

typedef std::array<SE_Type, 81> SE_VERTICAL_9x9;

class SE_Vertical_9x9 : public SE_Interface
{
	public:
		SE_Vertical_9x9() = default;
		virtual ~SE_Vertical_9x9() { ; }
		const SE_Type* GetStructuredElement (size_t& line) const {line = 9; return m_se.data();}

	private:
		const SE_VERTICAL_9x9 m_se = { 0, 0, 0, 1, 1, 1, 0, 0, 0,
					                   0, 0, 0, 1, 1, 1, 0, 0, 0,
			                           0, 0, 0, 1, 1, 1, 0, 0, 0,
			                           0, 0, 0, 1, 1, 1, 0, 0, 0,
			                           0, 0, 0, 1, 1, 1, 0, 0, 0,
			                           0, 0, 0, 1, 1, 1, 0, 0, 0,
			                           0, 0, 0, 1, 1, 1, 0, 0, 0,
			                           0, 0, 0, 1, 1, 1, 0, 0, 0,
			                           0, 0, 0, 1, 1, 1, 0, 0, 0 };
};