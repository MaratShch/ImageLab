#include "SE_Interface.hpp"
#include "SE_Square_3x3.hpp"
#include "SE_Square_5x5.hpp"
#include "SE_Square_7x7.hpp"
#include "SE_Square_9x9.hpp"
#include "SE_Vertical_3x3.hpp"
#include "SE_Vertical_5x5.hpp"
#include "SE_Vertical_7x7.hpp"
#include "SE_Vertical_9x9.hpp"
#include "SE_Horizontal_3x3.hpp"
#include "SE_Horizontal_5x5.hpp"
#include "SE_Horizontal_7x7.hpp"
#include "SE_Horizontal_9x9.hpp"
#include "SE_Cross_3x3.hpp"
#include "SE_Cross_5x5.hpp"
#include "SE_Cross_7x7.hpp"
#include "SE_Cross_9x9.hpp"
#include "SE_Frame_3x3.hpp"
#include "SE_Frame_5x5.hpp"
#include "SE_Frame_7x7.hpp"
#include "SE_Frame_9x9.hpp"
#include "SE_Ring_3x3.hpp"
#include "SE_Ring_5x5.hpp"
#include "SE_Ring_7x7.hpp"
#include "SE_Ring_9x9.hpp"
#include "SE_Disk_3x3.hpp"
#include "SE_Disk_5x5.hpp"
#include "SE_Disk_7x7.hpp"
#include "SE_Disk_9x9.hpp"
#include "SE_Dymond_3x3.hpp"
#include "SE_Dymond_5x5.hpp"
#include "SE_Dymond_7x7.hpp"
#include "SE_Dymond_9x9.hpp"


#ifdef _DEBUG
	static std::atomic<std::uint32_t> gInstanceCnt{};
#endif

SE_Interface* CreateSeInterface (const SeType& seType, const SeSize& seSize)
{
	SE_Interface* seInterface = nullptr;

	switch (seType)
	{
		case SE_TYPE_SQUARE:
			switch (seSize)
			{
				case SE_SIZE_3x3:
					seInterface = static_cast<SE_Interface*>(new SE_Square_3x3);
				break;

				case SE_SIZE_5x5:
					seInterface = static_cast<SE_Interface*>(new SE_Square_5x5);
				break;

				case SE_SIZE_7x7:
					seInterface = static_cast<SE_Interface*>(new SE_Square_7x7);
				break;

				case SE_SIZE_9x9:
				default:
					seInterface = static_cast<SE_Interface*>(new SE_Square_9x9);
				break;
			}
		break;

		case SE_TYPE_VERTICAL:
			switch (seSize)
			{
				case SE_SIZE_3x3:
					seInterface = static_cast<SE_Interface*>(new SE_Vertical_3x3);
				break;

				case SE_SIZE_5x5:
					seInterface = static_cast<SE_Interface*>(new SE_Vertical_5x5);
				break;

				case SE_SIZE_7x7:
					seInterface = static_cast<SE_Interface*>(new SE_Vertical_7x7);
				break;

				case SE_SIZE_9x9:
				default:
					seInterface = static_cast<SE_Interface*>(new SE_Vertical_9x9);
				break;
			}
		break;

		case SE_TYPE_HORIZONTAL:
			switch (seSize)
			{
				case SE_SIZE_3x3:
					seInterface = static_cast<SE_Interface*>(new SE_Horizontal_3x3);
				break;

				case SE_SIZE_5x5:
					seInterface = static_cast<SE_Interface*>(new SE_Horizontal_5x5);
				break;

				case SE_SIZE_7x7:
					seInterface = static_cast<SE_Interface*>(new SE_Horizontal_7x7);
				break;

				case SE_SIZE_9x9:
				default:
					seInterface = static_cast<SE_Interface*>(new SE_Horizontal_9x9);
				break;
			}
		break;

		case SE_TYPE_CROSS:
			switch (seSize)
			{
				case SE_SIZE_3x3:
					seInterface = static_cast<SE_Interface*>(new SE_Cross_3x3);
				break;

				case SE_SIZE_5x5:
					seInterface = static_cast<SE_Interface*>(new SE_Cross_5x5);
				break;

				case SE_SIZE_7x7:
					seInterface = static_cast<SE_Interface*>(new SE_Cross_7x7);
				break;

				case SE_SIZE_9x9:
				default:
					seInterface = static_cast<SE_Interface*>(new SE_Cross_9x9);
				break;
			}
		break;

		case SE_TYPE_FRAME:
			switch (seSize)
			{
				case SE_SIZE_3x3:
					seInterface = static_cast<SE_Interface*>(new SE_Frame_3x3);
				break;

				case SE_SIZE_5x5:
					seInterface = static_cast<SE_Interface*>(new SE_Frame_5x5);
				break;
	
				case SE_SIZE_7x7:
					seInterface = static_cast<SE_Interface*>(new SE_Frame_7x7);
				break;

				case SE_SIZE_9x9:
				default:
					seInterface = static_cast<SE_Interface*>(new SE_Frame_9x9);
				break;
			}
		break;

		case SE_TYPE_RING:
			switch (seSize)
			{
				case SE_SIZE_3x3:
					seInterface = static_cast<SE_Interface*>(new SE_Ring_3x3);
				break;

				case SE_SIZE_5x5:
					seInterface = static_cast<SE_Interface*>(new SE_Ring_5x5);
				break;

				case SE_SIZE_7x7:
					seInterface = static_cast<SE_Interface*>(new SE_Ring_7x7);
				break;

				case SE_SIZE_9x9:
				default:
					seInterface = static_cast<SE_Interface*>(new SE_Ring_9x9);
				break;
			}
		break;

		case SE_TYPE_DISK:
			switch (seSize)
			{
				case SE_SIZE_3x3:
					seInterface = static_cast<SE_Interface*>(new SE_Disk_3x3);
				break;

				case SE_SIZE_5x5:
					seInterface = static_cast<SE_Interface*>(new SE_Disk_5x5);
				break;

				case SE_SIZE_7x7:
					seInterface = static_cast<SE_Interface*>(new SE_Disk_7x7);
				break;

				case SE_SIZE_9x9:
				default:
					seInterface = static_cast<SE_Interface*>(new SE_Disk_9x9);
				break;
			}
		break;

		case SE_TYPE_DIAMOND:
			switch (seSize)
			{
				case SE_SIZE_3x3:
					seInterface = static_cast<SE_Interface*>(new SE_Dymond_3x3);
				break;

				case SE_SIZE_5x5:
					seInterface = static_cast<SE_Interface*>(new SE_Dymond_5x5);
				break;

				case SE_SIZE_7x7:
					seInterface = static_cast<SE_Interface*>(new SE_Dymond_7x7);
				break;

				case SE_SIZE_9x9:
				default:
					seInterface = static_cast<SE_Interface*>(new SE_Dymond_9x9);
				break;
			}
		break;

		default:
		break;
	}
#ifdef _DEBUG
	if (nullptr != seInterface)
		gInstanceCnt++;
#endif
	return seInterface;
}


void DeleteSeInterface (SE_Interface* seInterafce)
{
	if (nullptr != seInterafce)
	{
		delete seInterafce;
		seInterafce = nullptr;
#ifdef _DEBUG
		gInstanceCnt--;
#endif
	}
	return;
}