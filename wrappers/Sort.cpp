__aicore__ inline void Sort(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        AscendC::LocalTensor<uint8_t> &sharedTmpBuffer,
        int &offset_ptr,
        const int32_t data_length) {

    int32_t sort_repeat = 1;
    int32_t sort_src_elems = 32;
    int32_t sort_dst_elems = 64;
    int32_t sort_tmp_bytes = 4096;
    int32_t sort_off = (offset_ptr + 31) & ~31;
    offset_ptr = sort_off;

    AscendC::LocalTensor<float> sort_concat = sharedTmpBuffer.ReinterpretCast<float>()[offset_ptr / (int32_t)sizeof(float)];
    sort_concat.SetSize(sort_src_elems * (int32_t)sizeof(float));
    offset_ptr += sort_concat.GetSize();
    offset_ptr = (offset_ptr + 31) & ~31;

    AscendC::LocalTensor<uint32_t> sort_index = sharedTmpBuffer.ReinterpretCast<uint32_t>()[offset_ptr / (int32_t)sizeof(uint32_t)];
    sort_index.SetSize(sort_src_elems * (int32_t)sizeof(uint32_t));
    offset_ptr += sort_index.GetSize();
    offset_ptr = (offset_ptr + 31) & ~31;

    AscendC::LocalTensor<float> sort_dst = sharedTmpBuffer.ReinterpretCast<float>()[offset_ptr / (int32_t)sizeof(float)];
    sort_dst.SetSize(sort_dst_elems * (int32_t)sizeof(float));
    offset_ptr += sort_dst.GetSize();
    offset_ptr = (offset_ptr + 31) & ~31;

    AscendC::LocalTensor<float> sort_tmp = sharedTmpBuffer.ReinterpretCast<float>()[offset_ptr / (int32_t)sizeof(float)];
    sort_tmp.SetSize(sort_tmp_bytes);
    offset_ptr += sort_tmp_bytes;
    offset_ptr = (offset_ptr + 31) & ~31;

    for (int32_t i = 0; i < sort_src_elems; ++i) {
        sort_concat.SetValue(i, (i < data_length) ? input0.GetValue(i) : 0.0f);
        sort_index.SetValue(i, (uint32_t)i);
    }

    AscendC::Sort<float, true>(sort_dst, sort_concat, sort_index, sort_tmp, sort_repeat);

    for (int32_t i = 0; i < data_length; ++i) {
        int32_t si = (i < sort_src_elems) ? i : (sort_src_elems - 1);
        output.SetValue(i, sort_dst.GetValue(si * 2));
    }
}
