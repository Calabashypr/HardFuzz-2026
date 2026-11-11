
__aicore__ inline void LoadUnzipIndex_Mask1(const AscendC::LocalTensor<float> &output,
        const AscendC::LocalTensor<float> &input0,
        GM_ADDR srcIndexGm,
        AscendC::LocalTensor<uint8_t> &sharedTmpBuffer,
        int &offset_ptr,
        const int32_t data_length) {
    (void)GenerateRandomMask(((data_length > 0) ? ((data_length > 64) ? 64 : data_length) : 1));
    (void)sharedTmpBuffer; (void)offset_ptr;
    AscendC::GlobalTensor<int8_t> indexGlobal;
    indexGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(srcIndexGm), 512U);
#if __NPU_ARCH__ == 2002
    AscendC::LoadUnzipIndex(indexGlobal, 1U);
#else
    AscendC::LoadUnzipIndexCal(indexGlobal, 1U);
#endif
    AscendC::DataCopy(output, input0, (uint32_t)data_length);
}
