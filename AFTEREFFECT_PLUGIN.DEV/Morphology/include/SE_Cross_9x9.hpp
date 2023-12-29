#pragma once

#include "SE_Interface.hpp"

typedef std::array<SE_Type, 81> SE_CROSS_9x9;

class SE_Cross_9x9 : public SE_Interface
{
	public:
		SE_Cross_9x9() = default;
		virtual ~SE_Cross_9x9() { ; }
		const SE_Type* GetStructuredElement (size_t& line) const {line = 9; return m_se.data();}

	private:
		const SE_CROSS_9x9 m_se = { 0, 0, 0, 1, 1, 1, 0, 0, 0,
					                0, 0, 0, 1, 1, 1, 0, 0, 0,
			                        0, 0, 0, 1, 1, 1, 0, 0, 0,
			                        1, 1, 1, 1, 1, 1, 1, 1, 1,
			                        1, 1, 1, 1, 1, 1, 1, 1, 1,
			                        1, 1, 1, 1, 1, 1, 1, 1, 1,
			                        0, 0, 0, 1, 1, 1, 0, 0, 0,
			                        0, 0, 0, 1, 1, 1, 0, 0, 0,
			                        0, 0, 0, 1, 1, 1, 0, 0, 0 };
};