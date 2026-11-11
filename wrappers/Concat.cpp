__aicore__ inline void Concat(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        AscendC::LocalTensor<uint8_t> &sharedTmpBuffer,
        int &offset_ptr,
        const int32_t data_length) {

    const int32_t concat_repeat = 1;
    const int32_t concat_elems = concat_repeat * 16;
    int32_t concat_off = (offset_ptr + 31) & ~31;
    offset_ptr = concat_off;
    AscendC::LocalTensor<float> concat_dst = sharedTmpBuffer.ReinterpretCast<float>()[offset_ptr / (int32_t)sizeof(float)];
    concat_dst.SetSize(concat_elems * (int32_t)sizeof(float));
    offset_ptr += concat_dst.GetSize();
    offset_ptr = (offset_ptr + 31) & ~31;
    AscendC::LocalTensor<float> concat_tmp = sharedTmpBuffer.ReinterpretCast<float>()[offset_ptr / (int32_t)sizeof(float)];
    concat_tmp.SetSize(concat_elems * (int32_t)sizeof(float));
    offset_ptr += concat_tmp.GetSize();
    offset_ptr = (offset_ptr + 31) & ~31;
    for (int32_t i = 0; i < concat_elems; ++i) {
        concat_dst.SetValue(i, 0.0f);
        concat_tmp.SetValue(i, 0.0f);
    }
    for (int32_t i = 0; i < concat_elems; ++i) {
        concat_dst.SetValue(i, (i < data_length) ? input0.GetValue(i) : 0.0f);
    }
    AscendC::Concat<float>(concat_dst, input0, concat_tmp, concat_repeat);
    for (int32_t i = 0; i < data_length; ++i) {
        output.SetValue(i, concat_dst.GetValue(i % concat_elems));
    }
}
