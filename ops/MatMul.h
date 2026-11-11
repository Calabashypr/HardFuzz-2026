        Matmul<
        MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>,
        MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>,
        MatmulType<AscendC::TPosition::VECIN, CubeFormat::ND, float>,
        MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>
        > matmulObj;

        REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), matmulObj, &matmulTiling);
        matmulObj.SetTensorA(aGlobal);
        matmulObj.SetTensorB(bGlobal);
        matmulObj.SetBias(biasGlobal);
        matmulObj.IterateAll(cGlobal);
        matmulObj.End();
