__aicore__ inline void Extract(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        AscendC::LocalTensor<uint8_t> &sharedTmpBuffer,
        int &offset_ptr,
        const int32_t data_length) {

    int32_t ext_repeat = 1;
    int32_t ext_src_elems = 32;
    int32_t ext_sorted_elems = 64;
    int32_t ext_sort_tmp_bytes = 4096;
    int32_t ext_off = (offset_ptr + 31) & ~31;
    offset_ptr = ext_off;

    AscendC::LocalTensor<float> ext_concat = sharedTmpBuffer.ReinterpretCast<float>()[offset_ptr / (int32_t)sizeof(float)];
    ext_concat.SetSize(ext_src_elems * (int32_t)sizeof(float));
    offset_ptr += ext_concat.GetSize();
    offset_ptr = (offset_ptr + 31) & ~31;
    AscendC::LocalTensor<uint32_t> ext_index = sharedTmpBuffer.ReinterpretCast<uint32_t>()[offset_ptr / (int32_t)sizeof(uint32_t)];
    ext_index.SetSize(ext_src_elems * (int32_t)sizeof(uint32_t));
    offset_ptr += ext_index.GetSize();
    offset_ptr = (offset_ptr + 31) & ~31;
    AscendC::LocalTensor<float> ext_sorted = sharedTmpBuffer.ReinterpretCast<float>()[offset_ptr / (int32_t)sizeof(float)];
    ext_sorted.SetSize(ext_sorted_elems * (int32_t)sizeof(float));
    offset_ptr += ext_sorted.GetSize();
    offset_ptr = (offset_ptr + 31) & ~31;
    AscendC::LocalTensor<float> ext_sort_tmp = sharedTmpBuffer.ReinterpretCast<float>()[offset_ptr / (int32_t)sizeof(float)];
    ext_sort_tmp.SetSize(ext_sort_tmp_bytes);
    offset_ptr += ext_sort_tmp.GetSize();
    offset_ptr = (offset_ptr + 31) & ~31;

    AscendC::LocalTensor<float> ext_dst_val = sharedTmpBuffer.ReinterpretCast<float>()[offset_ptr / (int32_t)sizeof(float)];
    ext_dst_val.SetSize(ext_src_elems * (int32_t)sizeof(float));
    offset_ptr += ext_dst_val.GetSize();
    offset_ptr = (offset_ptr + 31) & ~31;
    AscendC::LocalTensor<uint32_t> ext_dst_idx = sharedTmpBuffer.ReinterpretCast<uint32_t>()[offset_ptr / (int32_t)sizeof(uint32_t)];
    ext_dst_idx.SetSize(ext_src_elems * (int32_t)sizeof(uint32_t));
    offset_ptr += ext_dst_idx.GetSize();
    offset_ptr = (offset_ptr + 31) & ~31;

    for (int32_t i = 0; i < ext_src_elems; ++i) {
        ext_concat.SetValue(i, (i < data_length) ? input0.GetValue(i) : 0.0f);
        ext_index.SetValue(i, (uint32_t)i);
    }
    AscendC::Sort<float, true>(ext_sorted, ext_concat, ext_index, ext_sort_tmp, ext_repeat);

    AscendC::Extract<float>(ext_dst_val, ext_dst_idx, ext_sorted, ext_repeat);

    for (int32_t i = 0; i < data_length; ++i) {
        output.SetValue(i, ext_dst_val.GetValue(i));
    }
}
