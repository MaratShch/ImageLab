#pragma once

namespace ImageLabMemoryUtils
{

	enum class MemOwnedPolicy
	{
		MEM_POLICY_NORMAL,
		MEM_POLICY_PRIVATE_USAGE,
		MEM_POLICY_RELEASE_AND_FORGET
	};

}