#ifndef ML_DTYPES__NPY_2_COMPAT_H_
#define ML_DTYPES__NPY_2_COMPAT_H_

// This file vendors parts of npy_2_compat.h from NumPy 2.5 needed to compile
// with NumPy 2.0-2.4.
// When Python 3.11 is dropped one could instead depend on 2.5+ at build time.

#if NPY_API_VERSION < 0x00000016

/*
 * Backport of `NPY_DT_legacy_descriptor_proto` (and ABI fix for slot IDs).
 * This backport allows dtypes that are currently implemented as legacy
 * (i.e. have a kind, char, a character code, and only the byte-order parameter)
 * to work with only minor changes on NumPy 2.0+ but use any part of the new
 * DType API they want to.
 * This also will allow us to deprecate the weirder parts of it, i.e. cast
 * registration.
 * (Possibly the only remaining change may be poor `dtype=` printing in
 * arrays, which can be worked around.)
 */
/*
 * `NPY_2_4_API_VERSION` and `NPY_2_5_API_VERSION` may not be defined when
 * this header is vendored alongside an older `numpyconfig.h`.  Provide
 * fallback definitions so the rest of the backport can use named constants.
 */
 #ifndef NPY_2_4_API_VERSION
 #define NPY_2_4_API_VERSION 0x00000015
 #endif
 #ifndef NPY_2_5_API_VERSION
 #define NPY_2_5_API_VERSION 0x00000016
 #endif

 #if NPY_TARGET_VERSION < NPY_2_5_API_VERSION \
         && NPY_TARGET_VERSION >= NPY_2_0_API_VERSION

 #ifndef NPY_DT_legacy_descriptor_proto
 #define NPY_DT_legacy_descriptor_proto ((1 << 11) - 1)
 #endif

 #define _PyArrayInitDTypeMeta_FromSpec \
     (*(int (*)(PyArray_DTypeMeta *, PyArrayDTypeMeta_Spec *))PyArray_API[362])
 #undef PyArrayInitDTypeMeta_FromSpec

 static inline int PyArrayInitDTypeMeta_FromSpec(
         PyArray_DTypeMeta *DType, PyArrayDTypeMeta_Spec *spec)
 {
     PyArray_DescrProto *proto = NULL;
     if (spec->slots != NULL && spec->slots[0].slot == NPY_DT_legacy_descriptor_proto) {
         proto = (PyArray_DescrProto *)spec->slots[0].pfunc;
     }

 #if NPY_TARGET_VERSION < NPY_2_4_API_VERSION
     /*
      * In NumPy 2.4 the slot IDs ABI was accidentally changed, so we translate
      * them even if `NPY_DT_legacy_descriptor_proto` is unused. The translation
      * is idempotent.
      */
     PyType_Slot *slot = spec->slots;
     int bad_offset = (PyArray_RUNTIME_VERSION >= NPY_2_4_API_VERSION)
             ? (1 << 10) : (1 << 11);
     int good_offset = (PyArray_RUNTIME_VERSION >= NPY_2_4_API_VERSION)
             ? (1 << 11) : (1 << 10);
     while (slot->slot != 0 || slot->pfunc != NULL) {
         if (slot->slot >= bad_offset && slot->slot < bad_offset + 30) {
             slot->slot += good_offset - bad_offset;
         }
         slot++;
     }
 #endif

     if (proto == NULL || PyArray_RUNTIME_VERSION >= NPY_2_5_API_VERSION) {
         return _PyArrayInitDTypeMeta_FromSpec(DType, spec);
     }

 #if defined(Py_LIMITED_API)
     PyErr_SetString(PyExc_RuntimeError,
         "NPY_DT_legacy_descriptor_proto backport not supported in Python limited API");
     return -1;
 #else

     /*
      * Step 1: Register old-style with a garbage typeobj so that
      * _PyArray_MapPyTypeToDType does NOT add the auto-DTypeMeta to the
      * pytype-to-DType dict (it bails out on NPY_DT_is_legacy for non-generic
      * types), regardless of whether the real scalar subclasses np.generic.
      */
     PyArray_DescrProto new_proto = *proto;
     new_proto.typeobj = &PyBaseObject_Type;
     int typenum = PyArray_RegisterDataType(&new_proto);
     if (typenum < 0) {
         return -1;
     }

     /*
      * Step 2: Initialise the user's DType with new-style slots and casts.
      * type_num stays at -1 / 0 for now; we fix it in step 3.
      */
     PyArrayDTypeMeta_Spec new_spec = *spec;
     new_spec.slots = &spec->slots[1];  /* skip proto slot */
     if (_PyArrayInitDTypeMeta_FromSpec(DType, &new_spec) < 0) {
         return -1;
     }

     /*
      * Step 3: Steal the singleton descriptor and type_num from the legacy
      * registration.  Point the descriptor's Python type at the user's DType
      * and fix up its typeobj field (which we temporarily set to
      * PyBaseObject_Type in step 1).
      */
     PyArray_Descr *descr = PyArray_DescrFromType(typenum);
     if (descr == NULL) {
         return -1;
     }

     /* Save the auto-DTypeMeta so we can decref it after the swap. */
     PyObject *old_meta = (PyObject *)Py_TYPE(descr);

     DType->type_num = typenum;
     /* PyArray_DescrFromType returns a new reference; transfer ownership. */
     DType->singleton = descr;
     /*
      * Set the legacy flag (bit 0 == _NPY_DT_LEGACY_FLAG) so NumPy uses
      * legacy code paths (copyswap, ArrFuncs, etc.) where the new-style API
      * doesn't cover them yet.
      */
     DType->flags |= 1;

     /* Re-type the descriptor so it belongs to the user's DType class. */
     Py_INCREF(DType);
     Py_SET_TYPE(descr, (PyTypeObject *)(DType));
     Py_DECREF(old_meta);

     /*
      * Fix the descriptor's scalar-type field (it was set to
      * PyBaseObject_Type in step 1 by PyArray_RegisterDataType copying
      * proto->typeobj).
      */
     Py_INCREF(proto->typeobj);
     Py_XDECREF(descr->typeobj);
     descr->typeobj = proto->typeobj;

     /*
      * Initialize legacy ArrFuncs from the descriptor prototype.
      */
     if (proto->f != NULL) {
         PyArray_ArrFuncs *f = _PyDataType_GetArrFuncs(descr);
         /*
          * Preserve ArrFuncs that were explicitly set via the new API slots
          * (step 2), and fill missing ones from the legacy prototype.
          * getitem/setitem always come from the legacy descriptor path.
          */
          if (proto->f->getitem != NULL) {
             f->getitem = proto->f->getitem;
         }
         if (proto->f->setitem != NULL) {
             f->setitem = proto->f->setitem;
         }
 #define NPY_PROTO_FILL_IF_NULL(FIELD) \
         if (f->FIELD == NULL) { \
             f->FIELD = proto->f->FIELD; \
         }
         NPY_PROTO_FILL_IF_NULL(copyswap);
         NPY_PROTO_FILL_IF_NULL(copyswapn);
         NPY_PROTO_FILL_IF_NULL(compare);
         NPY_PROTO_FILL_IF_NULL(argmax);
         NPY_PROTO_FILL_IF_NULL(dotfunc);
         NPY_PROTO_FILL_IF_NULL(scanfunc);
         NPY_PROTO_FILL_IF_NULL(fromstr);
         NPY_PROTO_FILL_IF_NULL(nonzero);
         NPY_PROTO_FILL_IF_NULL(fill);
         NPY_PROTO_FILL_IF_NULL(fillwithscalar);
         NPY_PROTO_FILL_IF_NULL(scalarkind);
         NPY_PROTO_FILL_IF_NULL(argmin);
 #undef NPY_PROTO_FILL_IF_NULL
         for (int i = 0; i < NPY_NSORTS; i++) {
             f->sort[i] = proto->f->sort[i];
             f->argsort[i] = proto->f->argsort[i];
         }
     }
 #endif  /* Py_LIMITED_API */
     return 0;
 }
 #endif


#endif

#endif  // ML_DTYPES__NPY_2_COMPAT_H_
