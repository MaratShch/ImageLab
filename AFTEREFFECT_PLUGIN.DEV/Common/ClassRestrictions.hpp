#pragma once

#define CLASS_NON_COPYABLE(TypeName)            \
TypeName(TypeName const&) = delete;             \
TypeName& operator = (TypeName const&) = delete

#define CLASS_NON_MOVABLE(TypeName)             \
TypeName(TypeName &&) = delete;                 \
TypeName& operator = (TypeName&&) = delete
