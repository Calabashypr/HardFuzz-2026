__aicore__ inline void AdjustSoftMaxRes(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        const AscendC::LocalTensor<float> &input1,
        const int32_t data_length) {

    AscendC::DataCopy(const_cast<AscendC::LocalTensor<float>&>(output), input0, data_length);

    AscendC::SoftMaxShapeInfo shape;
    shape.srcM = 1;
    shape.srcK = (uint32_t)data_length;
    shape.oriSrcM = 1;
    shape.oriSrcK = (uint32_t)data_length;

    const uint32_t from_bits = 0x3f800000U;
    const float to_val = 0.0f;
    AscendC::AdjustSoftMaxRes<float, float>(
        const_cast<AscendC::LocalTensor<float>&>(output),
        input1,
        from_bits,
        to_val,
        shape);
}
