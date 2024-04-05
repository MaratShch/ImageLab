#ifndef __IMAGELAB2_CLASS_RESTRICTIONS_PROPERTIES__
#define __IMAGELAB2_CLASS_RESTRICTIONS_PROPERTIES__

#ifndef CLASS_NON_COPYABLE
 #define CLASS_NON_COPYABLE(TypeName)            \
 TypeName(TypeName const&) = delete;             \
 TypeName& operator = (TypeName const&) = delete
#endif

#ifndef CLASS_NON_MOVABLE
 #define CLASS_NON_MOVABLE(TypeName)             \
 TypeName(TypeName &&) = delete;                 \
 TypeName& operator = (TypeName&&) = delete
#endif

#endif /*__IMAGELAB2_CLASS_RESTRICTIONS_PROPERTIES__ */