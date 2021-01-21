#include "LutHelper.hpp"
#include <thread>
#include <mutex>
#include <memory>


std::mutex gHelperProtectMutex;
static CACHE_ALIGN LutHelper gLutHelper (defaultLutHelperSize);


void InitLutHelper(const size_t& capacity)
{
	if (capacity > defaultLutHelperSize)
	{
		std::lock_guard<std::mutex>lk(gHelperProtectMutex);
		gLutHelper.reserve(capacity);
	}
	return;
}


void DisposeAllLUTs(void)
{
	std::lock_guard<std::mutex>lk(gHelperProtectMutex);

	for (auto& pLut : gLutHelper)
	{
		if (nullptr != pLut)
		{
			delete pLut;
			pLut = nullptr;
		}
	}

	return;
}


LutIdx addToLut (const std::string& lutName)
{
	LutIdx _v_pos = -1;
	LutIdx _f_pos = -1;

	std::lock_guard<std::mutex>lk(gHelperProtectMutex);

	/* search LUT from existed LUT' list */
	for (auto& pLut : gLutHelper)
	{
		_v_pos++;

		if (nullptr != pLut && pLut->GetLutName() == lutName)
		{
			_f_pos = _v_pos;
			pLut->increaseRefCnt();
			break;
		}
	}

	/* LUT doesn't found, let's add it */
	if (-1 == _f_pos)
	{
		CubeLUT* pNewLut = new CubeLUT;
		if (CubeLUT::OK == pNewLut->LoadCubeFile(lutName))
		{
			_v_pos = -1;
			for (auto& pLut : gLutHelper)
			{
				_v_pos++;
				if (pLut == nullptr)
				{
					_f_pos = _v_pos;
					gLutHelper[_f_pos] = pNewLut;
					break;
				}
			}
		}
	}

	/* Vector to small - just make resize (will be implemented later) */
	/* ... */

	return _f_pos;
}


LutObjHndl getLut(const LutIdx& idx)
{
	std::lock_guard<std::mutex>lk(gHelperProtectMutex);
	return (idx >= 0 && gLutHelper.size() > idx) ? gLutHelper[idx] : nullptr;
}


void removeLut(const LutIdx& idx)
{
	if (idx >= 0 && gLutHelper.size() > idx)
	{
		std::lock_guard<std::mutex>lk(gHelperProtectMutex);
		LutObjHndl pCubeLUT = gLutHelper[idx];

		if (nullptr != pCubeLUT)
		{
			const uint32_t refCnt = pCubeLUT->decreseRefCnt();
			if (0u == refCnt)
			{
				delete pCubeLUT;
				gLutHelper[idx] = pCubeLUT = nullptr;
			}
		}
	}

	return;
}
