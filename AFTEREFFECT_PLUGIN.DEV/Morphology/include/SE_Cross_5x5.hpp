#pragma once

#include "SE_Interface.hpp"

typedef std::array<SE_Type, 25> SE_CROSS_5x5;

class SE_Cross_5x5 : public SE_Interface
{
	public:
		SE_Cross_5x5() = default;
		virtual ~SE_Cross_5x5() { ; }
		const SE_Type* GetStructuredElement (size_t& line) const {line = 5; return m_se.data();}

	private:
		const SE_CROSS_5x5 m_se = { 0, 0, 1, 0, 0,
					                0, 0, 1, 0, 0,
			                        1, 1, 1, 1, 1,
			                        0, 0, 1, 0, 0,
			                        0, 0, 1, 0, 0 };
};