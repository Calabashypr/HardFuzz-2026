__aicore__ inline void Sort32_Mask2(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        AscendC::LocalTensor<uint8_t> &tmp,
        int &offset_ptr,
        const int32_t data_length) {
    int32_t s32_repeat = (data_length + 31) / 32;
    int32_t s32_src_elems = s32_repeat * 32;
    int32_t s32_dst_elems = s32_repeat * 64;
    AscendC::LocalTensor<float> s32_src = tmp.ReinterpretCast<float>()[offset_ptr / (int32_t)sizeof(float)];
    s32_src.SetSize(s32_src_elems * (int32_t)sizeof(float));
    offset_ptr += s32_src.GetSize();
    offset_ptr = (offset_ptr + 31) & ~31;
    AscendC::LocalTensor<uint32_t> s32_idx = tmp.ReinterpretCast<uint32_t>()[offset_ptr / (int32_t)sizeof(uint32_t)];
    s32_idx.SetSize(s32_src_elems * (int32_t)sizeof(uint32_t));
    offset_ptr += s32_idx.GetSize();
    offset_ptr = (offset_ptr + 31) & ~31;
    AscendC::LocalTensor<float> s32_dst = tmp.ReinterpretCast<float>()[offset_ptr / (int32_t)sizeof(float)];
    s32_dst.SetSize(s32_dst_elems * (int32_t)sizeof(float));
    offset_ptr += s32_dst.GetSize();
    offset_ptr = (offset_ptr + 31) & ~31;
    for (int32_t i = 0; i < s32_src_elems; ++i) {
        s32_src.SetValue(i, (i < data_length) ? input0.GetValue(i) : 0.0f);
        s32_idx.SetValue(i, (uint32_t)i);
    }
    AscendC::Sort32<float>(s32_dst, s32_src, s32_idx, s32_repeat);
    for (int32_t i = 0; i < data_length; ++i) {
        output.SetValue(i, s32_dst.GetValue(i * 2));
    }
}
