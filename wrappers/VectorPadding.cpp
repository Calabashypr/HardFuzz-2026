__aicore__ inline void VectorPadding(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        const int32_t data_length) {

    AscendC::printf("[DBG] VectorPadding enter len=%d in0=%f\n", data_length, input0.GetValue(0));
    AscendC::VectorPadding(output, input0, (uint8_t)0, false, (uint32_t)data_length);
    AscendC::printf("[DBG] VectorPadding exit out0=%f\n", output.GetValue(0));
}
