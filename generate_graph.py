import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, Set, Tuple, List, Optional

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from tools.log_utils import important, setup_logger

log = setup_logger("generate_graph")

operation_set = [
    "And",
    "Or",
    "Exp",
    "Sqrt",
    "Reciprocal",
    "Ln",
    "Rsqrt",
    "Relu",
    "Input_node",
    "ConstOne",
    "Add",
    "Mul",
    "Sub",
    "Div",
    "Max",
    "Min",
    "Abs",
    "Adds",
    "Muls",
    "Maxs",
    "Mins",
    "LeakyRelu",
    "Axpy",
    "ReduceMax",
    "Not",
    "AddRelu",
    "Acos",
    "SubRelu",
    "Acosh",
    "MulAddDst",
    "FusedMulAdd",
    "ScalarGetCountOfValue",
    "ScalarCountLeadingZero",
    "ScalarCast",
    "CountBitsCntSameAsSignBit",
    "ScalarGetSFFValue",
    "ShiftLeft",
    "ShiftRight",
    "Asin",
    "Asinh",
    "Atan",
    "Atanh",
    "Ceil",
    "ClampMax",
    "ClampMin",
    "Cos",
    "Cosh",
    "Cast",
    "Digamma",
    "Erf",
    "Erfc",
    "Floor",
    "Fmod",
    "Frac",
    "Lgamma",
    "Log",
    "Power",
    "Round",
    "Sign",
    "Sin",
    "Sinh",
    "Tan",
    "Tanh",
    "Trunc",
    "Xor",
    "ReduceMin",
    "ReduceSum",
    "Duplicate",
    "Brcb",
    "BilinearInterpolation",
    "AddReluCast",
    "AddDeqRelu",
    "SubReluCast",
    "MulCast",
    "WholeReduceMax",
    "Sort32",
    "MrgSort",
    "GetMrgSortResult",
    "GetReduceMaxMinCount",
    "WholeReduceMin",
    "WholeReduceSum",
    "BlockReduceMax",
    "BlockReduceMin",
    "BlockReduceSum",
    "PairReduceSum",
    "RepeatReduceSum",
    "CreateVecIndex",
    "VectorPadding",
    "Compare",
    "Select",
    "ProposalConcat",
    "CompareScalar",
    "GatherMask",
    "AdjustSoftMaxRes",
    "FasterGelu",
    "FasterGeluV2",
    "GeGLU",
    "GeLU",
    "LogSoftMax",
    "SoftmaxFlash",
    "SoftmaxFlashV2",
    "SoftmaxFlashV3",
    "ReGlu",
    "Sigmoid",
    "Silu",
    "SimpleSoftMax",
    "SoftMax",
    "GetAccVal",
    "SoftmaxGrad",
    "SoftmaxGradFront",
    "Swish",
    "SwiGLU",
    "DataCopyPad",
    "SetPadValue",
    "InitConstValue",
    "LoadUnzipIndex",
    "LoadDataUnzip",
    "LoadDataWithTranspose",
    "LoadDataWithSparse",
    "SetFmatrix",
    "SetLoadDataBoundary",
    "SetLoadDataRepeat",
    "SetLoadDataPaddingValue",
    "SetFixPipeConfig",
    "SetFixpipeNz2ndFlag",
    "SetFixpipePreQuantFlag",
    "SetFixPipeClipRelu",
    "SetFixPipeAddr",
    "SetAippFunctions",
    "LoadImageToLocal",
    "BatchNorm",
    "DeepNorm",
    "GroupNorm",
    "LayerNorm",
    "LayerNormGrad",
    "LayerNormGradBeta",
    "Normalize",
    "ReduceXorSum",
    "Mean",
    "RmsNorm",
    "WelfordFinalize",
    "WelfordUpdate",
    "TopK",
    "Sort",
    "Extract",
    "Concat",
    "Arange",
    "Broadcast",
    "Fill",
    "Transpose",
    "TransData",
    "Pad",
    "UnPad",
    "DropOut",
    "Gatherb",
    "Scatter",
    "CompareCmpMask",
    "GetCmpMask",
    "ReduceMean",
    "ReduceAny",
    "ReduceAll",
    "ReduceProd",
    "AscendAntiQuant",
    "AscendDequant",
    "AscendQuant",
    "ReduceSum2",
    "ReduceMax2",
    "ReduceMin2",
    "Exp_Mask1",
    "Exp_Mask2",
    "Ln_Mask1",
    "Ln_Mask2",
    "Abs_Mask1",
    "Abs_Mask2",
    "Reciprocal_Mask1",
    "Reciprocal_Mask2",
    "Sqrt_Mask1",
    "Sqrt_Mask2",
    "Rsqrt_Mask1",
    "Rsqrt_Mask2",
    "Not_Mask1",
    "Not_Mask2",
    "Relu_Mask1",
    "Relu_Mask2",
    "Add_Mask1",
    "Add_Mask2",
    "Sub_Mask1",
    "Sub_Mask2",
    "Mul_Mask1",
    "Mul_Mask2",
    "Div_Mask1",
    "Div_Mask2",
    "Max_Mask1",
    "Max_Mask2",
    "Min_Mask1",
    "Min_Mask2",
    "BilinearInterpolation_Mask1",
    "BilinearInterpolation_Mask2",
    "And_Mask1",
    "And_Mask2",
    "Or_Mask1",
    "Or_Mask2",
    "AddRelu_Mask1",
    "AddRelu_Mask2",
    "AddReluCast_Mask1",
    "AddReluCast_Mask2",
    "AddDeqRelu_Mask1",
    "AddDeqRelu_Mask2",
    "SubRelu_Mask1",
    "SubRelu_Mask2",
    "SubReluCast_Mask1",
    "SubReluCast_Mask2",
    "MulAddDst_Mask1",
    "MulAddDst_Mask2",
    "MulCast_Mask1",
    "MulCast_Mask2",
    "FusedMulAdd_Mask1",
    "FusedMulAdd_Mask2",
    "FusedMulAddRelu_Mask1",
    "FusedMulAddRelu_Mask2",
    "Adds_Mask1",
    "Adds_Mask2",
    "Muls_Mask1",
    "Muls_Mask2",
    "Maxs_Mask1",
    "Maxs_Mask2",
    "Mins_Mask1",
    "Mins_Mask2",
    "ShiftLeft_Mask1",
    "ShiftLeft_Mask2",
    "ShiftRight_Mask1",
    "ShiftRight_Mask2",
    "LeakyRelu_Mask1",
    "LeakyRelu_Mask2",
    "Axpy_Mask1",
    "Axpy_Mask2",
    "Compare_Mask1",
    "Compare_Mask2",
    "CompareCmpMask_Mask1",
    "CompareCmpMask_Mask2",
    "CompareScalar_Mask1",
    "CompareScalar_Mask2",
    "ScalarGetCountOfValue_Mask1",
    "ScalarGetCountOfValue_Mask2",
    "ScalarCountLeadingZero_Mask1",
    "ScalarCountLeadingZero_Mask2",
    "ScalarCast_Mask1",
    "ScalarCast_Mask2",
    "CountBitsCntSameAsSignBit_Mask1",
    "CountBitsCntSameAsSignBit_Mask2",
    "ScalarGetSFFValue_Mask1",
    "ScalarGetSFFValue_Mask2",
    "ToBfloat16_Mask1",
    "ToBfloat16_Mask2",
    "ToFloat_Mask1",
    "ToFloat_Mask2",
    "Select_Mask1",
    "Select_Mask2",
    "GatherMask_Mask1",
    "GatherMask_Mask2",
    "Cast_Mask1",
    "Cast_Mask2",
    "CastDeq_Mask1",
    "CastDeq_Mask2",
    "ReduceMax_Mask1",
    "ReduceMax_Mask2",
    "ReduceMin_Mask1",
    "ReduceMin_Mask2",
    "ReduceSum_Mask1",
    "ReduceSum_Mask2",
    "WholeReduceMax_Mask1",
    "WholeReduceMax_Mask2",
    "WholeReduceMin_Mask1",
    "WholeReduceMin_Mask2",
    "WholeReduceSum_Mask1",
    "WholeReduceSum_Mask2",
    "BlockReduceMax_Mask1",
    "BlockReduceMax_Mask2",
    "BlockReduceMin_Mask1",
    "BlockReduceMin_Mask2",
    "BlockReduceSum_Mask1",
    "BlockReduceSum_Mask2",
    "PairReduceSum_Mask1",
    "PairReduceSum_Mask2",
    "RepeatReduceSum_Mask1",
    "RepeatReduceSum_Mask2",
    "DataCopyPad_Mask1",
    "DataCopyPad_Mask2",
    "SetPadValue_Mask1",
    "SetPadValue_Mask2",
    "InitConstValue_Mask1",
    "InitConstValue_Mask2",
    "LoadData_Mask1",
    "LoadData_Mask2",
    "LoadUnzipIndex_Mask1",
    "LoadUnzipIndex_Mask2",
    "LoadDataUnzip_Mask1",
    "LoadDataUnzip_Mask2",
    "LoadDataWithTranspose_Mask1",
    "LoadDataWithTranspose_Mask2",
    "LoadDataWithSparse_Mask1",
    "LoadDataWithSparse_Mask2",
    "SetFmatrix_Mask1",
    "SetFmatrix_Mask2",
    "SetLoadDataBoundary_Mask1",
    "SetLoadDataBoundary_Mask2",
    "SetLoadDataRepeat_Mask1",
    "SetLoadDataRepeat_Mask2",
    "SetLoadDataPaddingValue_Mask1",
    "SetLoadDataPaddingValue_Mask2",
    "Fixpipe_Mask1",
    "Fixpipe_Mask2",
    "SetFixPipeConfig_Mask1",
    "SetFixPipeConfig_Mask2",
    "SetFixpipeNz2ndFlag_Mask1",
    "SetFixpipeNz2ndFlag_Mask2",
    "SetFixpipePreQuantFlag_Mask1",
    "SetFixpipePreQuantFlag_Mask2",
    "SetFixPipeClipRelu_Mask1",
    "SetFixPipeClipRelu_Mask2",
    "SetFixPipeAddr_Mask1",
    "SetFixPipeAddr_Mask2",
    "SetAippFunctions_Mask1",
    "SetAippFunctions_Mask2",
    "LoadImageToLocal_Mask1",
    "LoadImageToLocal_Mask2",
    "Duplicate_Mask1",
    "Duplicate_Mask2",
    "CreateVecIndex_Mask1",
    "CreateVecIndex_Mask2",
    "VectorPadding_Mask1",
    "VectorPadding_Mask2",
    "SetMaskCount_Mask1",
    "SetMaskCount_Mask2",
    "SetMaskNorm_Mask1",
    "SetMaskNorm_Mask2",
    "SetVectorMask_Mask1",
    "SetVectorMask_Mask2",
    "ResetMask_Mask1",
    "ResetMask_Mask2",
    "DataCopy_Mask1",
    "DataCopy_Mask2",
    "GetCmpMask_Mask1",
    "GetCmpMask_Mask2",
    "GetAccVal_Mask1",
    "GetAccVal_Mask2",
    "GetReduceMaxMinCount_Mask1",
    "GetReduceMaxMinCount_Mask2",
    "ProposalConcat_Mask1",
    "ProposalConcat_Mask2",
    "Sort32_Mask1",
    "Sort32_Mask2",
    "MrgSort_Mask1",
    "MrgSort_Mask2",
    "GetMrgSortResult_Mask1",
    "GetMrgSortResult_Mask2",
    "Gatherb_Mask1",
    "Gatherb_Mask2",
    "Scatter_Mask1",
    "Scatter_Mask2",
    "MatMul",
]

special_ops_list = ['Ln', 'Sqrt', 'Rsqrt', 'Acos', 'Acosh']
special_ops_alias_to_operation_set = {n: n for n in special_ops_list}
special_ops_in_operation_set = set(special_ops_alias_to_operation_set.values())

modules = {}
modules[0] = 2
modules[1] = 2
modules[2] = 1
modules[3] = 1
modules[4] = 1
modules[5] = 1
modules[6] = 1
modules[7] = 1
modules[8] = 0
modules[9] = 0
modules[10] = 2
modules[11] = 2
modules[12] = 2
modules[13] = 2
modules[14] = 2
modules[15] = 2
modules[16] = 1
modules[17] = 2
modules[18] = 2
modules[19] = 2
modules[20] = 2
modules[21] = 2
modules[22] = 2
modules[23] = 1
modules[24] = 1
modules[25] = 2
modules[26] = 1
modules[27] = 2
modules[28] = 1
modules[29] = 2
modules[30] = 2
modules[31] = 1
modules[32] = 1
modules[33] = 1
modules[34] = 1
modules[35] = 1

modules[36] = 2
modules[37] = 2
modules[38] = 1
modules[39] = 1
modules[40] = 1
modules[41] = 1
modules[42] = 1
modules[43] = 2
modules[44] = 2
modules[45] = 1
modules[46] = 1
modules[47] = 1
modules[48] = 2
modules[49] = 1
modules[50] = 1
modules[51] = 1
modules[52] = 2
modules[53] = 1
modules[54] = 1
modules[55] = 1
modules[56] = 2
modules[57] = 1
modules[58] = 1
modules[59] = 1
modules[60] = 1
modules[61] = 1
modules[62] = 1
modules[63] = 1
modules[64] = 2
modules[65] = 3
modules[66] = 3
modules[67] = 1
modules[68] = 1
modules[69] = 2
modules[70] = 2
modules[71] = 2
modules[72] = 2
modules[73] = 2
modules[74] = 1
modules[75] = 1
modules[76] = 1
modules[77] = 1
modules[78] = 1
modules[79] = 1
modules[80] = 1
modules[81] = 1
modules[82] = 1
modules[83] = 2
modules[84] = 2
modules[85] = 2
modules[86] = 1
modules[87] = 2
modules[88] = 2
modules[89] = 2
modules[90] = 1
modules[91] = 1
modules[92] = 2
modules[93] = 1
modules[94] = 1
modules[95] = 1
modules[96] = 1
modules[97] = 1
modules[98] = 2
modules[99] = 1
modules[100] = 1
modules[101] = 3
modules[102] = 1
modules[103] = 0
modules[104] = 1
modules[105] = 1
modules[106] = 1
modules[107] = 1
modules[108] = 2
modules[109] = 2
modules[110] = 1
modules[111] = 2
modules[112] = 1
modules[113] = 1
modules[114] = 1
modules[115] = 1
modules[116] = 1
modules[117] = 1
modules[118] = 1
modules[119] = 1
modules[120] = 1
modules[121] = 1
modules[122] = 1
modules[123] = 1
modules[124] = 1
modules[125] = 1
modules[126] = 1
modules[127] = 1
modules[128] = 1
modules[129] = 1
modules[130] = 3
modules[131] = 2
modules[132] = 2
modules[133] = 2
modules[134] = 2
modules[135] = 2
modules[136] = 2
modules[137] = 2
modules[138] = 1
modules[139] = 2
modules[140] = 2
modules[141] = 2
modules[142] = 1
modules[143] = 1
modules[144] = 1
modules[145] = 1
modules[146] = 1
modules[147] = 1
modules[148] = 2
modules[149] = 1
modules[150] = 1
modules[151] = 1
modules[152] = 1
modules[153] = 1
modules[154] = 1
modules[155] = 1
modules[156] = 0
modules[157] = 1
modules[158] = 0
modules[159] = 1
modules[160] = 1
modules[161] = 1
modules[162] = 1
modules[163] = 2
modules[164] = 1
modules[165] = 1
modules[166] = 2
modules[167] = 1
modules[174] = 2
modules[175] = 1
modules[176] = 1
modules[177] = 1
modules[178] = 1
modules[179] = 1
modules[180] = 1
modules[181] = 1
modules[182] = 1
modules[183] = 1
modules[184] = 1
modules[185] = 1
modules[186] = 1
modules[187] = 1
modules[188] = 1
modules[189] = 1
modules[190] = 1
modules[191] = 2
modules[192] = 2
modules[193] = 2
modules[194] = 2
modules[195] = 2
modules[196] = 2
modules[197] = 2
modules[198] = 2
modules[199] = 2
modules[200] = 2
modules[201] = 2
modules[202] = 2
modules[203] = 2
modules[204] = 2
modules[205] = 2
modules[206] = 2
modules[207] = 2
modules[208] = 2
modules[209] = 2
modules[210] = 2
modules[211] = 2
modules[212] = 2
modules[213] = 2
modules[214] = 2
modules[215] = 2
modules[216] = 2
modules[217] = 2
modules[218] = 2
modules[219] = 2
modules[220] = 1
modules[221] = 1
modules[222] = 1
modules[223] = 1
modules[224] = 1
modules[225] = 1
modules[226] = 2
modules[227] = 2
modules[228] = 1
modules[229] = 1
modules[230] = 1
modules[231] = 1
modules[232] = 1
modules[233] = 1
modules[234] = 2
modules[235] = 2
modules[236] = 2
modules[237] = 2
modules[238] = 2
modules[239] = 2
modules[240] = 2
modules[241] = 2
modules[242] = 1
modules[243] = 1
modules[244] = 1
modules[245] = 1
modules[246] = 1
modules[247] = 1
modules[248] = 1
modules[249] = 1
modules[250] = 1
modules[251] = 1
modules[252] = 1
modules[253] = 1
modules[254] = 1
modules[255] = 1
modules[256] = 2
modules[257] = 2
modules[258] = 2
modules[259] = 2
modules[260] = 1
modules[261] = 1
modules[262] = 1
modules[263] = 1
modules[264] = 1
modules[265] = 1
modules[266] = 1
modules[267] = 1
modules[268] = 1
modules[269] = 1
modules[270] = 1
modules[271] = 1
modules[272] = 1
modules[273] = 1
modules[274] = 1
modules[275] = 1
modules[276] = 1
modules[277] = 1
modules[278] = 1
modules[279] = 1
modules[280] = 1
modules[281] = 1
modules[282] = 1
modules[283] = 1
modules[284] = 1
modules[285] = 1
modules[286] = 1
modules[287] = 1
modules[288] = 1
modules[289] = 1
modules[290] = 1
modules[291] = 1
modules[292] = 1
modules[293] = 1
modules[294] = 1
modules[295] = 1
modules[296] = 1
modules[297] = 1
modules[298] = 1
modules[299] = 1
modules[300] = 1
modules[301] = 1
modules[302] = 1
modules[303] = 1
modules[304] = 1
modules[305] = 1
modules[306] = 1
modules[307] = 1
modules[308] = 1
modules[309] = 1
modules[310] = 1
modules[311] = 1
modules[312] = 1
modules[313] = 1
modules[314] = 1
modules[315] = 1
modules[316] = 1
modules[317] = 1
modules[318] = 1
modules[319] = 1
modules[320] = 1
modules[321] = 1
modules[322] = 1
modules[323] = 1
modules[324] = 1
modules[325] = 1

modules[326] = 1
modules[327] = 1
modules[328] = 1
modules[329] = 1
modules[330] = 1
modules[331] = 1
modules[332] = 1
modules[333] = 1
modules[334] = 1
modules[335] = 1
modules[336] = 1
modules[337] = 1
modules[338] = 1
modules[339] = 1
modules[340] = 1
modules[341] = 1
modules[342] = 1
modules[343] = 1
modules[344] = 1
modules[345] = 1
modules[346] = 1
modules[347] = 1
modules[348] = 1
modules[349] = 1
modules[350] = 1
modules[351] = 1
modules[352] = 1
modules[353] = 1
modules[354] = 1
modules[355] = 1
modules[356] = 1
modules[357] = 1
modules[358] = 1
modules[359] = 1
modules[360] = 2

class GeneratedGraph:
    def __init__(self, num_nodes: int):
        self.num_nodes = num_nodes
        self.adjacency: Dict[int, Set[int]] = {i: set() for i in range(num_nodes)}
        self.edge_tags: Dict[Tuple[int, int], Set[str]] = {}
        self.cycle_nodes: Set[int] = set()
        self.cycles: List[Tuple[int, int]] = []
        self.conditions: List[Tuple[int, int]] = []

        self.node_op: Dict[int, str] = {}
        self.cond_entries: Dict[int, Tuple[int, int]] = {}
        self.cond_exits: Dict[int, int] = {}
        self.loop_entries: Set[int] = set()
        self.loop_exits: Dict[int, int] = {}

    def add_edge(self, u: int, v: int, tag: Optional[str] = None) -> None:
        if u == v:
            return
        if not (0 <= u < self.num_nodes and 0 <= v < self.num_nodes):
            return
        self.adjacency[u].add(v)
        key = (u, v)
        if key not in self.edge_tags:
            self.edge_tags[key] = set()
        if tag:
            self.edge_tags[key].add(tag)

    def tag_edge(self, u: int, v: int, tag: str) -> None:
        key = (u, v)
        if key not in self.edge_tags:
            self.edge_tags[key] = set()
        self.edge_tags[key].add(tag)

    def out_degree(self, u: int) -> int:
        return len(self.adjacency[u])

    def append_node(self) -> int:

        n = self.num_nodes
        self.num_nodes += 1
        self.adjacency[n] = set()
        return n

    def edges(self) -> List[Tuple[int, int]]:
        out = []
        for u, vs in self.adjacency.items():
            for v in vs:
                out.append((u, v))
        return out

_RANDOM_POOL_EXCLUDED = frozenset({"Input_node", "ConstOne", "Fmod"})

_STRUCTURE_OPS = frozenset({"while_loop", "if-else-condition"})

class MCMCSamplerState:

    def __init__(
        self,
        counts: Optional[Dict[str, int]] = None,
        current_op: Optional[str] = None,
        current_ops: Optional[Dict[object, Optional[str]]] = None,
    ) -> None:
        self.counts: Dict[str, int] = {}
        for op, count in (counts or {}).items():
            value = int(count)
            if value < 0:
                raise ValueError(f"MCMC counts must be nonnegative: {op}={value}")
            self.counts[str(op)] = value
        self.current_ops: Dict[int, Optional[str]] = {}
        for arity, op in (current_ops or {}).items():
            if op is not None and not isinstance(op, str):
                raise ValueError("MCMC chain state must be a string or null")
            self.current_ops[int(arity)] = op

        if current_op is not None:
            self.current_ops.setdefault(1, str(current_op))

    def select(self, candidates: List[str], tau: float, arity: Optional[int] = None,
               rng: Optional[random.Random] = None) -> str:

        if not candidates:
            raise ValueError("Empty MCMC candidate pool")
        if tau <= 0 or not math.isfinite(tau):
            raise ValueError(f"mcmc_tau must be finite and positive; got {tau}")

        pick = rng or random
        k = int(arity) if arity is not None else 1
        current_op = self.current_ops.get(k)

        if current_op not in candidates:
            current_op = None

        if current_op is None:
            current_op = pick.choice(candidates)

        proposal = pick.choice(candidates)
        current_count = self.counts.get(current_op, 0)
        proposal_count = self.counts.get(proposal, 0)
        count_delta = current_count - proposal_count

        acceptance = 1.0 if count_delta >= 0 else math.exp(count_delta / tau)
        selected = proposal if pick.random() < acceptance else current_op

        self.current_ops[k] = selected
        self.counts[selected] = self.counts.get(selected, 0) + 1
        return selected

    def to_dict(self) -> Dict[str, object]:
        return {
            "version": 2,
            "current_ops": {str(k): v for k, v in sorted(self.current_ops.items())},
            "counts": dict(sorted(self.counts.items())),
        }

    @classmethod
    def from_dict(cls, data: object) -> "MCMCSamplerState":
        if not isinstance(data, dict):
            raise ValueError("MCMC state must be a JSON object")
        counts = data.get("counts", {})
        if not isinstance(counts, dict):
            raise ValueError("MCMC counts must be an object")
        current_ops = data.get("current_ops")
        if current_ops is not None and not isinstance(current_ops, dict):
            raise ValueError("MCMC current_ops must be an object")
        current_op = data.get("current_op")
        if current_op is not None and not isinstance(current_op, str):
            raise ValueError("MCMC current_op must be a string or null")
        return cls(counts=counts, current_op=current_op, current_ops=current_ops)

def load_mcmc_state(path: str) -> MCMCSamplerState:
    state_path = Path(path)
    if not state_path.exists():
        return MCMCSamplerState()
    return MCMCSamplerState.from_dict(
        json.loads(state_path.read_text(encoding="utf-8"))
    )

def save_mcmc_state(path: str, state: MCMCSamplerState) -> None:
    state_path = Path(path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = state_path.with_name(state_path.name + ".tmp")
    temporary_path.write_text(
        json.dumps(state.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(state_path)

def decay_to_mcmc_tau(op_decay: float) -> float:
    decay = float(op_decay)
    if not 0.0 < decay < 1.0:
        raise ValueError("op_decay must be in (0, 1); prefer --mcmc-tau")
    return -1.0 / math.log(decay)

def get_op_to_module() -> Dict[str, int]:
    return {operation_set[i]: modules.get(i, 1) for i in range(len(operation_set))}

def _forward_topology_and_dominators(
    num_nodes: int,
    edges: List[Tuple[int, int]],
) -> Tuple[List[int], Dict[int, Set[int]]]:

    preds: Dict[int, Set[int]] = {i: set() for i in range(num_nodes)}
    succs: Dict[int, Set[int]] = {i: set() for i in range(num_nodes)}
    for u, v in edges:
        if u == v or not (0 <= u < num_nodes and 0 <= v < num_nodes):
            continue
        succs[u].add(v)
        preds[v].add(u)

    root = num_nodes
    succs[root] = {v for v in range(num_nodes) if not preds[v]}
    for v in succs[root]:
        preds[v].add(root)
    preds[root] = set()

    indegree = {v: len(preds[v]) for v in range(num_nodes + 1)}
    indegree[root] = 0
    queue: List[int] = [root]
    order: List[int] = []
    while queue:
        u = queue.pop()
        order.append(u)
        for w in succs.get(u, set()):
            indegree[w] -= 1
            if indegree[w] == 0:
                queue.append(w)
    for v in range(num_nodes + 1):
        if v not in order:
            order.append(v)

    dom: Dict[int, Set[int]] = {v: set(order) for v in order}
    dom[root] = {root}
    changed = True
    while changed:
        changed = False
        for v in order:
            if v == root:
                continue
            ps = preds[v]
            if not ps:
                new_dom: Set[int] = {v}
            else:
                inter: Optional[Set[int]] = None
                for p in ps:
                    inter = set(dom[p]) if inter is None else (inter & dom[p])
                    if not inter:
                        break
                new_dom = {v} | (inter or set())
            if new_dom != dom[v]:
                dom[v] = new_dom
                changed = True

    topo_order = [v for v in order if v != root]
    return topo_order, dom

def build_module_to_ops(ops: Set[str]) -> Dict[int, List[str]]:

    op_to_mod = get_op_to_module()
    grouped: Dict[int, List[str]] = {}
    for op in sorted(ops):
        if op in op_to_mod:
            grouped.setdefault(op_to_mod[op], []).append(op)
    return grouped

def replace_ops_with_subset(
    g: GeneratedGraph,
    allowed_ops: Set[str],
    rng: Optional[random.Random] = None,
) -> int:

    if not allowed_ops:
        return 0
    pick = (rng or random).choice
    op_to_mod = get_op_to_module()
    module_pool = build_module_to_ops(allowed_ops)
    replaced = 0
    for n in range(g.num_nodes):
        op = g.node_op.get(n)
        if op is None or op in _STRUCTURE_OPS or op in allowed_ops:
            continue
        mod = op_to_mod.get(op)
        if mod is None:

            mod = 2
        candidates = [c for c in module_pool.get(mod, []) if c != op]
        if not candidates:
            continue
        g.node_op[n] = pick(candidates)
        replaced += 1
    return replaced

def load_ops_from_file(path: str) -> str:
    parts: List[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            for tok in line.replace(",", " ").split():
                tok = tok.strip()
                if tok:
                    parts.append(tok)
    if not parts:
        raise ValueError(f"Empty interface list file: {path}")
    return ",".join(parts)

def parse_ops_csv(raw: Optional[str]) -> Optional[Set[str]]:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    return {p.strip() for p in s.split(",") if p.strip()}

def resolve_allowed_ops(
    ops_csv: Optional[str] = None,
    ops_file: Optional[str] = None,
    *,
    include_special_ops: bool = True,
) -> Optional[Set[str]]:
    spec_parts: List[str] = []
    if ops_file:
        spec_parts.append(load_ops_from_file(ops_file))
    if ops_csv:
        raw = str(ops_csv).strip()
        if raw.startswith("@"):
            spec_parts.append(load_ops_from_file(raw[1:].strip()))
        else:
            spec_parts.append(raw)
    if not spec_parts:
        return None
    requested = parse_ops_csv(",".join(spec_parts))
    if requested is None:
        return None
    unknown = requested - set(operation_set)
    if unknown:
        raise ValueError(
            "Unknown interfaces: " + ", ".join(sorted(unknown)) + "; use --list-ops to list interfaces"
        )
    allowed = requested - set(_RANDOM_POOL_EXCLUDED)
    if not allowed:
        raise ValueError("Empty interface pool (Input_node/ConstOne/Fmod are excluded)")
    if not include_special_ops:
        allowed -= special_ops_in_operation_set
    return allowed

# Build control flow, then select arity-compatible interfaces.
def generate_directed_graph(
    num_nodes: int = 35,
    branch_prob: float = 0.2,
    max_cycles: int = 2,
    max_outdegree: int = 3,
    cycle_len_min: int = 3,
    cycle_len_max: int = 6,
    max_conds: int = 2,
    cond_len_min: int = 3,
    cond_len_max: int = 6,
    op_decay: Optional[float] = None,
    seed: Optional[int] = None,
    nested_loop_prob: float = 0.5,
    nested_loop_depth_max: int = 5,
    include_special_ops: bool = True,
    allowed_ops: Optional[Set[str]] = None,
    mcmc_tau: float = 2.8,
    mcmc_state: Optional[MCMCSamplerState] = None,
    operator_selection: str = "mcmc",
    graph_structure: str = "structured",
    ops_budget: bool = False,
) -> GeneratedGraph:
    if operator_selection not in {"mcmc", "random"}:
        raise ValueError(
            "operator_selection must be 'mcmc' or 'random'; "
            f"got {operator_selection!r}"
        )
    if graph_structure not in {"structured", "random-dag"}:
        raise ValueError(
            "graph_structure must be 'structured' or 'random-dag'; "
            f"got {graph_structure!r}"
        )

    if seed is None:
        struct_rng = random.Random()
        op_rng = random.Random()
    else:
        struct_rng = random.Random(seed)
        op_rng = random.Random(f"operator-selection:{seed}")

    def in_op_pool(op: str) -> bool:
        return allowed_ops is None or op in allowed_ops

    g = GeneratedGraph(num_nodes)

    for i in range(num_nodes - 1):
        g.add_edge(i, i + 1)

    branch_budget = max(0, round(num_nodes * branch_prob))
    branches_added = 0
    for i in range(num_nodes - 2):
        if branches_added >= branch_budget:
            break
        if struct_rng.random() < branch_prob and g.out_degree(i) < max_outdegree:
            upper = min(num_nodes - 1, i + 5)
            lower = i + 2
            if upper >= lower:
                candidates = [j for j in range(lower, upper + 1) if j not in g.adjacency[i]]
                if candidates:
                    j = struct_rng.choice(candidates)
                    g.add_edge(i, j)
                    branches_added += 1

    cycles_target = (
        min(2, max(1, max_cycles)) if graph_structure == "structured" else 0
    )
    attempts = 0
    cycles_added = 0

    existing_cycles = []
    while cycles_added < cycles_target and attempts < 100:
        attempts += 1

        length_candidates = list(range(cycle_len_min, cycle_len_max + 1))
        length_weights = [(l - cycle_len_min + 1) ** 2 for l in length_candidates]
        length = struct_rng.choices(length_candidates, weights=length_weights, k=1)[0]

        max_start = max(1, num_nodes - length - 2)
        if max_start < 1:
            continue
        a = struct_rng.randint(1, max_start)
        b = min(num_nodes - 2, a + length - 1)
        if b <= a:
            continue

        overlaps = False
        for (ea, eb) in existing_cycles:

            overlap_start = max(a, ea)
            overlap_end = min(b, eb)
            if overlap_start < overlap_end:
                overlap_len = overlap_end - overlap_start + 1
                new_len = b - a + 1
                if overlap_len > new_len * 0.5:
                    overlaps = True
                    break

        if overlaps:
            continue

        if g.out_degree(b) >= max_outdegree:
            continue
        if (b, a) in g.edge_tags or a not in g.adjacency or b not in g.adjacency:
            continue

        g.add_edge(b, a, tag="cycle")
        for k in range(a, b):
            g.tag_edge(k, k + 1, "cycle")
        g.tag_edge(a - 1, a, "loop_entry")
        g.tag_edge(b, b + 1, "loop_exit")
        for k in range(a, b + 1):
            g.cycle_nodes.add(k)
        g.cycles.append((a, b))
        existing_cycles.append((a, b))
        cycles_added += 1

    nested_new_cycles: List[Tuple[int, int]] = []

    sorted_cycles = sorted(g.cycles, key=lambda x: x[1] - x[0], reverse=True)

    for idx, (a, b) in enumerate(sorted_cycles):
        if b - a < 2:
            continue

        body_width = max(0, b - a + 1)

        theoretical_cap = max(1, body_width // 3)
        depth_cap = min(nested_loop_depth_max, theoretical_cap)

        if depth_cap <= 0:
            continue

        is_large_cycle = idx < 2 and body_width >= (cycle_len_min + cycle_len_max) // 2
        if is_large_cycle:

            effective_prob = min(0.95, nested_loop_prob * 1.5)

            candidates = list(range(1, depth_cap + 1))
            weights = [k ** 2 for k in candidates]
        else:

            effective_prob = nested_loop_prob
            candidates = list(range(1, depth_cap + 1))
            weights = [k for k in candidates]

        if struct_rng.random() < effective_prob:

            total_w = sum(weights)
            r = struct_rng.uniform(0.0, total_w)
            acc = 0.0
            depth = candidates[-1]
            for k, w in zip(candidates, weights):
                acc += w
                if r <= acc:
                    depth = k
                    break
        else:
            depth = 0
        cur_a, cur_b = a, b
        for layer_idx in range(depth):

            if cur_b - cur_a < 2:
                break

            available_width = cur_b - cur_a - 1
            if available_width < 3:
                break

            inner_start_min = cur_a + 1
            inner_start_max = max(inner_start_min, cur_b - 2)
            if inner_start_min > inner_start_max:
                break

            if layer_idx >= 2:

                mid_point = (inner_start_min + inner_start_max) // 2
                candidates = list(range(inner_start_min, min(mid_point + 1, inner_start_max + 1)))
                if candidates:
                    inner_a = struct_rng.choice(candidates)
                else:
                    inner_a = struct_rng.randint(inner_start_min, inner_start_max)
            else:
                inner_a = struct_rng.randint(inner_start_min, inner_start_max)

            max_len_allowed = max(3, min(cycle_len_max, (cur_b - inner_a)))
            if max_len_allowed < cycle_len_min:
                break

            if layer_idx >= 2:

                preferred_max = min(max_len_allowed, cycle_len_min + 2)
                inner_len = struct_rng.randint(cycle_len_min, preferred_max)
            else:
                inner_len = struct_rng.randint(cycle_len_min, max_len_allowed)

            inner_b = min(cur_b - 1, inner_a + inner_len - 1)
            if inner_b <= inner_a:
                break

            if (inner_b, inner_a) in g.edge_tags:

                break

            g.add_edge(inner_b, inner_a, tag="cycle")
            for k in range(inner_a, inner_b):
                g.tag_edge(k, k + 1, "cycle")
            g.tag_edge(inner_a - 1, inner_a, "loop_entry")
            g.tag_edge(inner_b, inner_b + 1, "loop_exit")
            for k in range(inner_a, inner_b + 1):
                g.cycle_nodes.add(k)
            nested_new_cycles.append((inner_a, inner_b))

            cur_a, cur_b = inner_a, inner_b

    g.cycles.extend(nested_new_cycles)

    cond_target = (
        max(0, min(max_conds, num_nodes // 20 + 1))
        if graph_structure == "structured"
        else 0
    )
    cond_attempts = 0
    cond_added = 0
    while cond_added < cond_target and cond_attempts < 100:
        cond_attempts += 1

        s = struct_rng.randint(1, num_nodes - 4)

        cond_len = struct_rng.randint(cond_len_min, cond_len_max)

        offset_max = min(cond_len, (num_nodes - 3) - s)
        if offset_max <= 2:
            continue
        b = s + struct_rng.randint(2, offset_max)
        a = s + 1
        if b == a:
            continue

        if g.out_degree(s) >= max_outdegree:
            continue

        if b not in g.adjacency[s]:
            g.add_edge(s, b, tag="cond_branch")
        g.tag_edge(s, a, "cond_branch")

        g.tag_edge(s - 1, s, "cond_entry")

        m = max(a, b) + 1
        g.tag_edge(m, m + 1, "cond_exit")

        g.conditions.append((s, m))
        cond_added += 1

    for (a, b) in g.cycles:
        g.loop_entries.add(a)
        g.loop_exits[b] = a

    for (s, m) in g.conditions:
        succs = [v for v in g.adjacency[s] if "cond_branch" in g.edge_tags.get((s, v), set())]

        if len(succs) < 2:
            succs = sorted(list(g.adjacency[s]))[:2]
        if len(succs) >= 2:

            struct_rng.shuffle(succs)
            g.cond_entries[s] = (succs[0], succs[1])
            g.cond_exits[m] = s

    if ops_budget and graph_structure == "structured":
        struct_node_ids = set()
        for (u, v), tags in g.edge_tags.items():
            if "loop_entry" in tags:
                struct_node_ids.add(u)
            if "cond_entry" in tags:
                struct_node_ids.add(v)

        struct_node_ids.discard(0)
        for _ in range(len(struct_node_ids)):
            g.append_node()

    op_to_arity = {operation_set[i]: modules.get(i, 1) for i in range(len(operation_set))}
    pool_ops_all = (
        [op for op in operation_set if in_op_pool(op)]
        if allowed_ops is not None
        else list(operation_set)
    )
    if not pool_ops_all:
        raise ValueError("Empty interface pool; expand --ops or enable special interfaces")
    effective_tau = float(mcmc_tau)
    if operator_selection == "mcmc":
        effective_tau = (
            decay_to_mcmc_tau(op_decay) if op_decay is not None else float(mcmc_tau)
        )
        if effective_tau <= 0 or not math.isfinite(effective_tau):
            raise ValueError(f"mcmc_tau must be finite and positive; got {effective_tau}")
    sampler_state = mcmc_state if mcmc_state is not None else MCMCSamplerState()

    excluded_ops = set(_RANDOM_POOL_EXCLUDED)
    ops = [
        o for o in pool_ops_all
        if o not in excluded_ops
        and (include_special_ops or o not in special_ops_in_operation_set)
    ]
    if not ops:
        raise ValueError("No available candidates; adjust --ops or --disable-special-ops")

    avail_by_k: Dict[int, List[str]] = {}
    for op in ops:
        ar = int(op_to_arity.get(op, 1))
        avail_by_k.setdefault(ar, []).append(op)
    k_max = max(avail_by_k) if avail_by_k else 1

    structure_op: Dict[int, str] = {}
    for (u, v), tags in g.edge_tags.items():
        if "loop_entry" in tags:
            structure_op[u] = "while_loop"
        if "cond_entry" in tags:
            structure_op[v] = "if-else-condition"

    structure_op.pop(0, None)

    forward_edges = [(u, v) for u, vs in g.adjacency.items() for v in vs if u < v]
    topo_order, dom = _forward_topology_and_dominators(g.num_nodes, forward_edges)

    incoming: Dict[int, Set[int]] = {i: set() for i in range(g.num_nodes)}
    for u, vs in g.adjacency.items():
        for v in vs:
            incoming[v].add(u)

    def _has_structural_tag(u: int, v: int) -> bool:
        tags = g.edge_tags.get((u, v), set())
        return bool(tags & {"cycle", "loop_entry", "loop_exit",
                            "cond_entry", "cond_exit", "cond_branch"})

    def _counts_as_data(u: int, v: int) -> bool:

        tags = g.edge_tags.get((u, v), set())
        return ("cycle" not in tags) and ("loop_entry" not in tags)

    for n in topo_order:

        if n in structure_op and n != 0:
            g.node_op[n] = structure_op[n]
            continue

        n_incoming = set(incoming.get(n, set()))

        undeletable = sorted(
            u for u in n_incoming
            if _counts_as_data(u, n)
            and _has_structural_tag(u, n)
            and u not in structure_op
        )
        undeletable_set = set(undeletable)

        exits = sorted(
            u for u in n_incoming
            if _counts_as_data(u, n)
            and u not in structure_op
            and u not in undeletable_set
            and any(a <= u <= b < n for (a, b) in g.cycles)
        )

        kept = undeletable + exits[: max(0, k_max - len(undeletable))]
        kept_set = set(kept)

        p_v: List[int] = []
        for p in sorted(dom.get(n, set())):
            if not (0 <= p < g.num_nodes and p < n):
                continue
            if p in structure_op or g.node_op.get(p) in (None, "while_loop", "if-else-condition"):
                continue
            if p in kept_set:
                continue
            tags = g.edge_tags.get((p, n), set())
            if tags and not _counts_as_data(p, n):
                continue
            if any(a <= p <= b < n for (a, b) in g.cycles):
                continue
            p_v.append(p)

        kept_n = len(kept)
        feasible = sorted(
            k for k in avail_by_k
            if max(1, kept_n) <= k <= kept_n + len(p_v)
        )

        src_share = len(avail_by_k.get(0, ())) / max(1, len(ops))
        if (
            n != 0
            and kept_n == 0
            and avail_by_k.get(0)
            and op_rng.random() < src_share
        ):
            chosen: List[int] = []
            candidate_ops = avail_by_k[0]
            selected_arity = 0
        elif feasible:
            k = op_rng.choice(feasible)
            chosen = kept + op_rng.sample(p_v, k - kept_n)
            candidate_ops = avail_by_k[k]
            selected_arity = k
        else:

            positive = sorted(k for k in avail_by_k if k >= 1)
            if not positive:
                positive = [0]
            sufficient = [kk for kk in positive if kk >= kept_n]
            k = sufficient[0] if sufficient else positive[-1]
            if k <= kept_n:
                chosen = kept[:k]
            else:
                chosen = kept + op_rng.sample(p_v, min(k - kept_n, len(p_v)))
                if len(chosen) < k:
                    filled = set(chosen) | kept_set
                    for p in range(n - 1, -1, -1):
                        if len(chosen) >= k:
                            break
                        if p in filled or p in structure_op:
                            continue
                        if g.node_op.get(p) in (None, "while_loop", "if-else-condition"):
                            continue
                        edge_tags = g.edge_tags.get((p, n), set())
                        if edge_tags and not _counts_as_data(p, n):
                            continue
                        chosen.append(p)
                        filled.add(p)
            candidate_ops = avail_by_k.get(k) or ops
            selected_arity = k
        chosen_set = set(chosen)

        for u in list(n_incoming):
            if not _counts_as_data(u, n):
                continue
            if u in structure_op or u in chosen_set:
                continue
            if _has_structural_tag(u, n):
                continue
            g.adjacency[u].discard(n)
            incoming[n].discard(u)
            g.edge_tags.pop((u, n), None)
        for p in chosen:
            if p not in incoming.get(n, set()):
                g.add_edge(p, n)
                incoming[n].add(p)

        if n in g.cycle_nodes or any(a <= n <= b for a, b in g.cycles) or any(s < n < m for s, m in g.conditions):
            candidate_ops = [op for op in candidate_ops if op != "MatMul"]
            if not candidate_ops:
                raise ValueError("MatMul needs a same-arity alternative inside loops/branches; expand --ops")

        if operator_selection == "mcmc":
            g.node_op[n] = sampler_state.select(
                candidate_ops, effective_tau, arity=selected_arity, rng=op_rng
            )
        else:
            g.node_op[n] = op_rng.choice(candidate_ops)

    if g.node_op.get(0) in (None, "while_loop", "if-else-condition"):
        g.node_op[0] = op_rng.choice(avail_by_k.get(1) or ops)

    return g

def to_dot(g: GeneratedGraph) -> str:
    lines = [
        "digraph G {",
        "  rankdir=LR;",
        '  graph [bgcolor="white", fontsize=32, fontname="Arial", pad="0.7", margin="0.5", nodesep="1.0", ranksep="1.2"];',
        '  node [shape=circle, style=filled, fillcolor=white, fontsize=32, fontname="Arial"];',
        '  edge [fontsize=30, fontname="Arial", penwidth=2.0];',
    ]
    for n in range(g.num_nodes):

        label_parts: List[str] = [str(n)]
        if n in g.node_op:
            label_parts.append(f"op:{g.node_op[n]}")
        if n in g.cond_entries:
            if_succ, else_succ = g.cond_entries[n]
            label_parts.append(f"if->{if_succ}")
            label_parts.append(f"else->{else_succ}")
        if n in g.cond_exits:
            label_parts.append(f"cond_entry:{g.cond_exits[n]}")
        if n in g.loop_entries:
            label_parts.append("loop_entry")
        if n in g.loop_exits:
            label_parts.append(f"loop_entry:{g.loop_exits[n]}")
        label = "\\n".join(label_parts)
        if n in g.cycle_nodes:
            lines.append(f"  {n} [fillcolor=lightcoral, color=red, label=\"{label}\"];")
        else:
            lines.append(f"  {n} [label=\"{label}\"];")
    for (u, v) in g.edges():
        tags = g.edge_tags.get((u, v), set())
        color = "black"

        if "cond_entry" in tags:
            color = "orange"
        elif "cond_exit" in tags:
            color = "purple"
        elif "loop_entry" in tags:
            color = "green"
        elif "loop_exit" in tags:
            color = "blue"
        elif "cycle" in tags:
            color = "red"
        elif "cond_branch" in tags:
            color = "gray"
        label = ",".join(sorted(tags)) if tags else ""
        if label:
            lines.append(f"  {u} -> {v} [color={color}, label=\"{label}\"];")
        else:
            lines.append(f"  {u} -> {v} [color={color}];")
    lines.append("}")
    return "\n".join(lines)

def to_json(g: GeneratedGraph) -> Dict:
    return {
        "num_nodes": g.num_nodes,
        "adjacency": {str(k): sorted(list(v)) for k, v in g.adjacency.items()},
        "edges": [
            {
                "from": u,
                "to": v,
                "tags": sorted(list(g.edge_tags.get((u, v), set())))
            }
            for (u, v) in g.edges()
        ],
        "cycle_nodes": sorted(list(g.cycle_nodes)),
        "cycles": [{"start": a, "end": b} for (a, b) in g.cycles],
        "conditions": [{"split": s, "merge": m} for (s, m) in g.conditions],
        "loop_entry_edges": [list(k) for k, t in g.edge_tags.items() if "loop_entry" in t],
        "loop_exit_edges": [list(k) for k, t in g.edge_tags.items() if "loop_exit" in t],
        "cond_entry_edges": [list(k) for k, t in g.edge_tags.items() if "cond_entry" in t],
        "cond_exit_edges": [list(k) for k, t in g.edge_tags.items() if "cond_exit" in t],
        "cond_branch_edges": [list(k) for k, t in g.edge_tags.items() if "cond_branch" in t],
        "nodes": [
            {
                "id": n,
                "op": g.node_op.get(n),
                "cond_if": g.cond_entries.get(n, (None, None))[0] if n in g.cond_entries else None,
                "cond_else": g.cond_entries.get(n, (None, None))[1] if n in g.cond_entries else None,
                "cond_entry_node": g.cond_exits.get(n),
                "loop_entry": n in g.loop_entries,
                "loop_entry_node": g.loop_exits.get(n),
            }
            for n in range(g.num_nodes)
        ],
    }

def main():
    parser = argparse.ArgumentParser(description="Generate structured computation graphs or random DAGs")
    parser.add_argument("--nodes", type=int, default=35, help="Number of nodes, or interface nodes with --ops-budget")
    parser.add_argument("--branch-prob", type=float, default=0.2)
    parser.add_argument("--max-cycles", type=int, default=2)
    parser.add_argument("--max-outdegree", type=int, default=3)
    parser.add_argument("--cycle-len-min", type=int, default=3)
    parser.add_argument("--cycle-len-max", type=int, default=6)
    parser.add_argument("--max-conds", type=int, default=2)
    parser.add_argument("--cond-len-min", type=int, default=3)
    parser.add_argument("--cond-len-max", type=int, default=6)
    parser.add_argument(
        "--operator-selection",
        choices=("mcmc", "random"),
        default="mcmc",
        help="Interface selection: mcmc (default) or uniform random",
    )
    parser.add_argument(
        "--graph-structure",
        choices=("structured", "random-dag"),
        default="structured",
        help="Graph structure: structured (loops/branches) or random-dag",
    )
    parser.add_argument(
        "--mcmc-tau",
        type=float,
        default=2.8,
        help="Temperature for exp(-count/tau), default 2.8",
    )
    parser.add_argument(
        "--mcmc-state",
        type=str,
        default=None,
        help="Persistent MCMC state file",
    )
    parser.add_argument(
        "--reset-mcmc-state",
        action="store_true",
        help="Start with an empty MCMC state",
    )
    parser.add_argument("--op-decay", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for independent topology and interface-selection random streams. "
             "The same seed shares structural generation across selection methods",
    )
    parser.add_argument(
        "--ops-budget",
        action="store_true",
        help="Count only interface nodes, excluding control-flow nodes; "
             "the interface count equals --nodes (structured graphs add compensation nodes; "
             "no change is needed for random-dag)",
    )
    parser.add_argument("--nested-loop-prob", type=float, default=0.5, help="Loop nesting probability")
    parser.add_argument("--nested-loop-depth-max", type=int, default=5, help="Maximum loop nesting depth")
    parser.add_argument("--disable-special-ops", action="store_true", help="Exclude special interfaces from the candidate pool")
    parser.add_argument(
        "--ops",
        type=str,
        default=None,
        help="Comma-separated interfaces, e.g. Add,Mul,Sub; use @ops.txt for a file; default: full pool",
    )
    parser.add_argument(
        "--ops-file",
        type=str,
        default=None,
        help="Interface list file (lines or commas; # comments); combined with --ops",
    )
    parser.add_argument("--list-ops", action="store_true", help="List interface names and exit")
    parser.add_argument("--dot", type=str, default=None)
    parser.add_argument("--json", type=str, default=None)

    args = parser.parse_args()
    if args.list_ops:
        for name in operation_set:
            print(name)
        return

    allowed_ops = resolve_allowed_ops(
        args.ops,
        args.ops_file,
        include_special_ops=(not args.disable_special_ops),
    )
    if args.reset_mcmc_state and not args.mcmc_state:
        parser.error("--reset-mcmc-state requires --mcmc-state")
    if args.operator_selection == "random" and (
        args.mcmc_state or args.reset_mcmc_state or args.op_decay is not None
    ):
        parser.error(
            "random selection does not accept --mcmc-state, --reset-mcmc-state or --op-decay"
        )
    sampler_state = (
        MCMCSamplerState()
        if args.reset_mcmc_state
        else load_mcmc_state(args.mcmc_state) if args.mcmc_state else MCMCSamplerState()
    )
    g = generate_directed_graph(
        num_nodes=args.nodes,
        branch_prob=args.branch_prob,
        max_cycles=args.max_cycles,
        max_outdegree=args.max_outdegree,
        cycle_len_min=args.cycle_len_min,
        cycle_len_max=args.cycle_len_max,
        max_conds=args.max_conds,
        cond_len_min=args.cond_len_min,
        cond_len_max=args.cond_len_max,
        op_decay=args.op_decay,
        seed=args.seed,
        nested_loop_prob=args.nested_loop_prob,
        nested_loop_depth_max=args.nested_loop_depth_max,
        include_special_ops=(not args.disable_special_ops),
        allowed_ops=allowed_ops,
        mcmc_tau=args.mcmc_tau,
        mcmc_state=sampler_state,
        operator_selection=args.operator_selection,
        graph_structure=args.graph_structure,
        ops_budget=args.ops_budget,
    )

    if args.operator_selection == "mcmc" and args.mcmc_state:
        save_mcmc_state(args.mcmc_state, sampler_state)

    if allowed_ops is not None:
        log.info(f"Interface pool ({len(allowed_ops)}): {', '.join(sorted(allowed_ops))}")

    important(
        f"Strategy: operator={args.operator_selection} graph={args.graph_structure}  "
        f"Nodes: {g.num_nodes}  Edges: {len(g.edges())}  "
        f"Loops: {len(g.cycles)}/{len(g.cycle_nodes)} nodes  Conditions: {len(g.conditions)}"
    )
    log.debug(f"Loop nodes: {sorted(list(g.cycle_nodes))}")
    log.debug("Example node labels:")
    for n in range(min(5, g.num_nodes)):
        extras = []
        if n in g.cond_entries:
            if_succ, else_succ = g.cond_entries[n]
            extras.append(f"if->{if_succ}, else->{else_succ}")
        if n in g.cond_exits:
            extras.append(f"cond_entry:{g.cond_exits[n]}")
        if n in g.loop_entries:
            extras.append("loop_entry")
        if n in g.loop_exits:
            extras.append(f"loop_entry:{g.loop_exits[n]}")
        log.debug(f"  Node {n}: op={g.node_op.get(n)}; {'; '.join(extras) if extras else 'no structural tags'}")

    if args.dot:
        dot_str = to_dot(g)
        with open(args.dot, "w", encoding="utf-8") as f:
            f.write(dot_str)
        important(f"DOT written: {args.dot}")

    if args.json:
        data = to_json(g)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        important(f"JSON written: {args.json}")

if __name__ == "__main__":
    main()
