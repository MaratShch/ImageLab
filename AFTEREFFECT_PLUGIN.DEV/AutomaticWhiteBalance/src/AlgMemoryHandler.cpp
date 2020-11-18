#include "AlgMemoryHandler.hpp"
#include "CompileTimeUtils.hpp"


CAlgMemHandler::CAlgMemHandler() noexcept
{
	m_memBytesSize = 0ul;
	m_p[0] = m_p[1] = nullptr;
}

CAlgMemHandler::~CAlgMemHandler() noexcept
{
	MemFree();
}

bool CAlgMemHandler::MemInit (const size_t& size) noexcept
{
	constexpr size_t CpuPageSize = 4096ul;
	bool bResult = true;

	/* check existed memory size and realloc for new size if required */
	if (m_memBytesSize < size)
	{
		const size_t alignedBytesSize = CreateAlignment(size, CpuPageSize);

		m_protect[0].lock();
		if (nullptr != m_p[0])
		{
			_aligned_free(m_p[0]);
			m_p[0] = nullptr;
		} /* if (nullptr != m_p[0]) */

		m_p[0] = _aligned_malloc(alignedBytesSize, CpuPageSize);
#ifdef _DEBUG
		memset(m_p[0], 0, alignedBytesSize);
#endif
		m_protect[0].unlock();

		m_protect[1].lock();
		if (nullptr != m_p[1])
		{
			_aligned_free(m_p[1]);
			m_p[1] = nullptr;
		} /* if (nullptr != m_p[1]) */

		m_p[1] = _aligned_malloc(alignedBytesSize, CpuPageSize);
#ifdef _DEBUG
		memset(m_p[1], 0, alignedBytesSize);
#endif
		m_protect[1].unlock();

		if (nullptr != m_p[0] && nullptr != m_p[1])
		{
			m_memBytesSize = alignedBytesSize;
			bResult = true;
		}
		else
		{
			bResult = false;
		}

	} /* if (m_memBytesSize < size) */

	return bResult;
}


void CAlgMemHandler::MemFree(void) noexcept
{
	m_memBytesSize = 0ul;

	m_protect[0].lock();
	m_protect[1].lock();

	if (nullptr != m_p[0])
	{
		_aligned_free(m_p[0]);
		m_p[0] = nullptr;
	}

	if (nullptr != m_p[1])
	{
		_aligned_free(m_p[1]);
		m_p[1] = nullptr;
	}

	m_protect[1].unlock();
	m_protect[0].unlock();

	return;
}
