#pragma once

#include "SE_Interface.hpp"

typedef std::array<SE_Type, 25> SE_SQUARE_5x5;

class SE_Square_5x5 : public SE_Interface
{
	public:
		SE_Square_5x5 () = default;
		virtual ~SE_Square_5x5() { ; }
		const SE_Type* GetStructuredElement(size_t& line) const { line = 5; return m_se.data(); }

	private:
		const SE_SQUARE_5x5 m_se = { 1, 1, 1, 1, 1,
					                 1, 1, 1, 1, 1,
					                 1, 1, 1, 1, 1,
						             1, 1, 1, 1, 1,
                                     1, 1, 1, 1, 1 };
};