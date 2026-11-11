__aicore__ inline void CreateVecIndex(const AscendC::LocalTensor<float> &output,
        const int32_t data_length) {

    AscendC::printf("[DBG] CreateVecIndex enter len=%d\n", data_length);
    AscendC::CreateVecIndex(output, 0.0f, (uint32_t)data_length);
    AscendC::printf("[DBG] CreateVecIndex exit out0=%f\n", output.GetValue(0));
}
