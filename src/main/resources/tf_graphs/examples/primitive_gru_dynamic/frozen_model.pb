
C
inputPlaceholder*
dtype0* 
shape:���������
.
RankConst*
dtype0*
value	B :
5
range/startConst*
dtype0*
value	B :
5
range/deltaConst*
dtype0*
value	B :
:
rangeRangerange/startRankrange/delta*

Tidx0
D
concat/values_0Const*
dtype0*
valueB"       
5
concat/axisConst*
dtype0*
value	B : 
U
concatConcatV2concat/values_0rangeconcat/axis*

Tidx0*
T0*
N
;
	transpose	Transposeinputconcat*
Tperm0*
T0
6
	rnn/ShapeShape	transpose*
out_type0*
T0
E
rnn/strided_slice/stackConst*
dtype0*
valueB:
G
rnn/strided_slice/stack_1Const*
dtype0*
valueB:
G
rnn/strided_slice/stack_2Const*
dtype0*
valueB:
�
rnn/strided_sliceStridedSlice	rnn/Shapernn/strided_slice/stackrnn/strided_slice/stack_1rnn/strided_slice/stack_2*
new_axis_mask *
Index0*

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask
c
9rnn/MultiRNNCellZeroState/GRUCellZeroState/ExpandDims/dimConst*
dtype0*
value	B : 
�
5rnn/MultiRNNCellZeroState/GRUCellZeroState/ExpandDims
ExpandDimsrnn/strided_slice9rnn/MultiRNNCellZeroState/GRUCellZeroState/ExpandDims/dim*

Tdim0*
T0
^
0rnn/MultiRNNCellZeroState/GRUCellZeroState/ConstConst*
dtype0*
valueB:
`
6rnn/MultiRNNCellZeroState/GRUCellZeroState/concat/axisConst*
dtype0*
value	B : 
�
1rnn/MultiRNNCellZeroState/GRUCellZeroState/concatConcatV25rnn/MultiRNNCellZeroState/GRUCellZeroState/ExpandDims0rnn/MultiRNNCellZeroState/GRUCellZeroState/Const6rnn/MultiRNNCellZeroState/GRUCellZeroState/concat/axis*

Tidx0*
T0*
N
g
6rnn/MultiRNNCellZeroState/GRUCellZeroState/zeros/ConstConst*
dtype0*
valueB 2        
�
0rnn/MultiRNNCellZeroState/GRUCellZeroState/zerosFill1rnn/MultiRNNCellZeroState/GRUCellZeroState/concat6rnn/MultiRNNCellZeroState/GRUCellZeroState/zeros/Const*
T0
e
;rnn/MultiRNNCellZeroState/GRUCellZeroState_1/ExpandDims/dimConst*
dtype0*
value	B : 
�
7rnn/MultiRNNCellZeroState/GRUCellZeroState_1/ExpandDims
ExpandDimsrnn/strided_slice;rnn/MultiRNNCellZeroState/GRUCellZeroState_1/ExpandDims/dim*

Tdim0*
T0
`
2rnn/MultiRNNCellZeroState/GRUCellZeroState_1/ConstConst*
dtype0*
valueB:
b
8rnn/MultiRNNCellZeroState/GRUCellZeroState_1/concat/axisConst*
dtype0*
value	B : 
�
3rnn/MultiRNNCellZeroState/GRUCellZeroState_1/concatConcatV27rnn/MultiRNNCellZeroState/GRUCellZeroState_1/ExpandDims2rnn/MultiRNNCellZeroState/GRUCellZeroState_1/Const8rnn/MultiRNNCellZeroState/GRUCellZeroState_1/concat/axis*

Tidx0*
T0*
N
i
8rnn/MultiRNNCellZeroState/GRUCellZeroState_1/zeros/ConstConst*
dtype0*
valueB 2        
�
2rnn/MultiRNNCellZeroState/GRUCellZeroState_1/zerosFill3rnn/MultiRNNCellZeroState/GRUCellZeroState_1/concat8rnn/MultiRNNCellZeroState/GRUCellZeroState_1/zeros/Const*
T0
e
;rnn/MultiRNNCellZeroState/GRUCellZeroState_2/ExpandDims/dimConst*
dtype0*
value	B : 
�
7rnn/MultiRNNCellZeroState/GRUCellZeroState_2/ExpandDims
ExpandDimsrnn/strided_slice;rnn/MultiRNNCellZeroState/GRUCellZeroState_2/ExpandDims/dim*

Tdim0*
T0
`
2rnn/MultiRNNCellZeroState/GRUCellZeroState_2/ConstConst*
dtype0*
valueB:
b
8rnn/MultiRNNCellZeroState/GRUCellZeroState_2/concat/axisConst*
dtype0*
value	B : 
�
3rnn/MultiRNNCellZeroState/GRUCellZeroState_2/concatConcatV27rnn/MultiRNNCellZeroState/GRUCellZeroState_2/ExpandDims2rnn/MultiRNNCellZeroState/GRUCellZeroState_2/Const8rnn/MultiRNNCellZeroState/GRUCellZeroState_2/concat/axis*

Tidx0*
T0*
N
i
8rnn/MultiRNNCellZeroState/GRUCellZeroState_2/zeros/ConstConst*
dtype0*
valueB 2        
�
2rnn/MultiRNNCellZeroState/GRUCellZeroState_2/zerosFill3rnn/MultiRNNCellZeroState/GRUCellZeroState_2/concat8rnn/MultiRNNCellZeroState/GRUCellZeroState_2/zeros/Const*
T0
e
;rnn/MultiRNNCellZeroState/GRUCellZeroState_3/ExpandDims/dimConst*
dtype0*
value	B : 
�
7rnn/MultiRNNCellZeroState/GRUCellZeroState_3/ExpandDims
ExpandDimsrnn/strided_slice;rnn/MultiRNNCellZeroState/GRUCellZeroState_3/ExpandDims/dim*

Tdim0*
T0
`
2rnn/MultiRNNCellZeroState/GRUCellZeroState_3/ConstConst*
dtype0*
valueB:
b
8rnn/MultiRNNCellZeroState/GRUCellZeroState_3/concat/axisConst*
dtype0*
value	B : 
�
3rnn/MultiRNNCellZeroState/GRUCellZeroState_3/concatConcatV27rnn/MultiRNNCellZeroState/GRUCellZeroState_3/ExpandDims2rnn/MultiRNNCellZeroState/GRUCellZeroState_3/Const8rnn/MultiRNNCellZeroState/GRUCellZeroState_3/concat/axis*

Tidx0*
T0*
N
i
8rnn/MultiRNNCellZeroState/GRUCellZeroState_3/zeros/ConstConst*
dtype0*
valueB 2        
�
2rnn/MultiRNNCellZeroState/GRUCellZeroState_3/zerosFill3rnn/MultiRNNCellZeroState/GRUCellZeroState_3/concat8rnn/MultiRNNCellZeroState/GRUCellZeroState_3/zeros/Const*
T0
e
;rnn/MultiRNNCellZeroState/GRUCellZeroState_4/ExpandDims/dimConst*
dtype0*
value	B : 
�
7rnn/MultiRNNCellZeroState/GRUCellZeroState_4/ExpandDims
ExpandDimsrnn/strided_slice;rnn/MultiRNNCellZeroState/GRUCellZeroState_4/ExpandDims/dim*

Tdim0*
T0
`
2rnn/MultiRNNCellZeroState/GRUCellZeroState_4/ConstConst*
dtype0*
valueB:
b
8rnn/MultiRNNCellZeroState/GRUCellZeroState_4/concat/axisConst*
dtype0*
value	B : 
�
3rnn/MultiRNNCellZeroState/GRUCellZeroState_4/concatConcatV27rnn/MultiRNNCellZeroState/GRUCellZeroState_4/ExpandDims2rnn/MultiRNNCellZeroState/GRUCellZeroState_4/Const8rnn/MultiRNNCellZeroState/GRUCellZeroState_4/concat/axis*

Tidx0*
T0*
N
i
8rnn/MultiRNNCellZeroState/GRUCellZeroState_4/zeros/ConstConst*
dtype0*
valueB 2        
�
2rnn/MultiRNNCellZeroState/GRUCellZeroState_4/zerosFill3rnn/MultiRNNCellZeroState/GRUCellZeroState_4/concat8rnn/MultiRNNCellZeroState/GRUCellZeroState_4/zeros/Const*
T0
8
rnn/Shape_1Shape	transpose*
out_type0*
T0
G
rnn/strided_slice_1/stackConst*
dtype0*
valueB: 
I
rnn/strided_slice_1/stack_1Const*
dtype0*
valueB:
I
rnn/strided_slice_1/stack_2Const*
dtype0*
valueB:
�
rnn/strided_slice_1StridedSlicernn/Shape_1rnn/strided_slice_1/stackrnn/strided_slice_1/stack_1rnn/strided_slice_1/stack_2*
new_axis_mask *
Index0*

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask
2
rnn/timeConst*
dtype0*
value	B : 
�
rnn/TensorArrayTensorArrayV3rnn/strided_slice_1*
dtype0*
clear_after_read(*/
tensor_array_namernn/dynamic_rnn/output_0*
dynamic_size( *
element_shape:
�
rnn/TensorArray_1TensorArrayV3rnn/strided_slice_1*
dtype0*
clear_after_read(*.
tensor_array_namernn/dynamic_rnn/input_0*
dynamic_size( *
element_shape:
I
rnn/TensorArrayUnstack/ShapeShape	transpose*
out_type0*
T0
X
*rnn/TensorArrayUnstack/strided_slice/stackConst*
dtype0*
valueB: 
Z
,rnn/TensorArrayUnstack/strided_slice/stack_1Const*
dtype0*
valueB:
Z
,rnn/TensorArrayUnstack/strided_slice/stack_2Const*
dtype0*
valueB:
�
$rnn/TensorArrayUnstack/strided_sliceStridedSlicernn/TensorArrayUnstack/Shape*rnn/TensorArrayUnstack/strided_slice/stack,rnn/TensorArrayUnstack/strided_slice/stack_1,rnn/TensorArrayUnstack/strided_slice/stack_2*
new_axis_mask *
Index0*

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask
L
"rnn/TensorArrayUnstack/range/startConst*
dtype0*
value	B : 
L
"rnn/TensorArrayUnstack/range/deltaConst*
dtype0*
value	B :
�
rnn/TensorArrayUnstack/rangeRange"rnn/TensorArrayUnstack/range/start$rnn/TensorArrayUnstack/strided_slice"rnn/TensorArrayUnstack/range/delta*

Tidx0
�
>rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3rnn/TensorArray_1rnn/TensorArrayUnstack/range	transposernn/TensorArray_1:1*
_class
loc:@transpose*
T0

rnn/while/EnterEnterrnn/time*
parallel_iterations *
is_constant( *
T0*$

frame_namernn/while/rnn/while/
�
rnn/while/Enter_1Enterrnn/TensorArray:1*
parallel_iterations *
is_constant( *
T0*$

frame_namernn/while/rnn/while/
�
rnn/while/Enter_2Enter0rnn/MultiRNNCellZeroState/GRUCellZeroState/zeros*
parallel_iterations *
is_constant( *
T0*$

frame_namernn/while/rnn/while/
�
rnn/while/Enter_3Enter2rnn/MultiRNNCellZeroState/GRUCellZeroState_1/zeros*
parallel_iterations *
is_constant( *
T0*$

frame_namernn/while/rnn/while/
�
rnn/while/Enter_4Enter2rnn/MultiRNNCellZeroState/GRUCellZeroState_2/zeros*
parallel_iterations *
is_constant( *
T0*$

frame_namernn/while/rnn/while/
�
rnn/while/Enter_5Enter2rnn/MultiRNNCellZeroState/GRUCellZeroState_3/zeros*
parallel_iterations *
is_constant( *
T0*$

frame_namernn/while/rnn/while/
�
rnn/while/Enter_6Enter2rnn/MultiRNNCellZeroState/GRUCellZeroState_4/zeros*
parallel_iterations *
is_constant( *
T0*$

frame_namernn/while/rnn/while/
T
rnn/while/MergeMergernn/while/Enterrnn/while/NextIteration*
T0*
N
Z
rnn/while/Merge_1Mergernn/while/Enter_1rnn/while/NextIteration_1*
T0*
N
Z
rnn/while/Merge_2Mergernn/while/Enter_2rnn/while/NextIteration_2*
T0*
N
Z
rnn/while/Merge_3Mergernn/while/Enter_3rnn/while/NextIteration_3*
T0*
N
Z
rnn/while/Merge_4Mergernn/while/Enter_4rnn/while/NextIteration_4*
T0*
N
Z
rnn/while/Merge_5Mergernn/while/Enter_5rnn/while/NextIteration_5*
T0*
N
Z
rnn/while/Merge_6Mergernn/while/Enter_6rnn/while/NextIteration_6*
T0*
N
�
rnn/while/Less/EnterEnterrnn/strided_slice_1*
parallel_iterations *
is_constant(*
T0*$

frame_namernn/while/rnn/while/
F
rnn/while/LessLessrnn/while/Mergernn/while/Less/Enter*
T0
.
rnn/while/LoopCondLoopCondrnn/while/Less
l
rnn/while/SwitchSwitchrnn/while/Mergernn/while/LoopCond*"
_class
loc:@rnn/while/Merge*
T0
r
rnn/while/Switch_1Switchrnn/while/Merge_1rnn/while/LoopCond*$
_class
loc:@rnn/while/Merge_1*
T0
r
rnn/while/Switch_2Switchrnn/while/Merge_2rnn/while/LoopCond*$
_class
loc:@rnn/while/Merge_2*
T0
r
rnn/while/Switch_3Switchrnn/while/Merge_3rnn/while/LoopCond*$
_class
loc:@rnn/while/Merge_3*
T0
r
rnn/while/Switch_4Switchrnn/while/Merge_4rnn/while/LoopCond*$
_class
loc:@rnn/while/Merge_4*
T0
r
rnn/while/Switch_5Switchrnn/while/Merge_5rnn/while/LoopCond*$
_class
loc:@rnn/while/Merge_5*
T0
r
rnn/while/Switch_6Switchrnn/while/Merge_6rnn/while/LoopCond*$
_class
loc:@rnn/while/Merge_6*
T0
;
rnn/while/IdentityIdentityrnn/while/Switch:1*
T0
?
rnn/while/Identity_1Identityrnn/while/Switch_1:1*
T0
?
rnn/while/Identity_2Identityrnn/while/Switch_2:1*
T0
?
rnn/while/Identity_3Identityrnn/while/Switch_3:1*
T0
?
rnn/while/Identity_4Identityrnn/while/Switch_4:1*
T0
?
rnn/while/Identity_5Identityrnn/while/Switch_5:1*
T0
?
rnn/while/Identity_6Identityrnn/while/Switch_6:1*
T0
�
!rnn/while/TensorArrayReadV3/EnterEnterrnn/TensorArray_1*
parallel_iterations *
is_constant(*
T0*$

frame_namernn/while/rnn/while/
�
#rnn/while/TensorArrayReadV3/Enter_1Enter>rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
parallel_iterations *
is_constant(*
T0*$

frame_namernn/while/rnn/while/
�
rnn/while/TensorArrayReadV3TensorArrayReadV3!rnn/while/TensorArrayReadV3/Enterrnn/while/Identity#rnn/while/TensorArrayReadV3/Enter_1*
dtype0
�
/rnn/multi_rnn_cell/cell_0/gru_cell/gates/kernelConst*
dtype0*�
value�B�	"���/�Ť���g�tO�?��V8�?�X>��?x�9���?N�u�Ծ�
���I�?ϸ7Pg��?��M����?PC�g��?���п�q�?��?pٚO�^�?^DYx�?�?Z�,��?1�����9 ��������}�?3$��,�ֿ��|ڂ���'���&��I��l��?E0����?�1}�;��?l�5yz�ܿ�YT�ƿ�-�y,�?&���8�?�4{�`�ܿ�\�?��?C�ҵ��?{M_�d޿�Ej�ۿYvJHyX��f�~�?E��Z�j��c�������{ �V�ֿ�zof�?����0g�?*-n�ȿ[���Q>ֿ�"�U*�������?����V��?H��,�u��̨.V�eܿ�����W�?![^cK�?n���Iw�?j��Ϳ���a/
����R8�?���&�	�?
z
4rnn/multi_rnn_cell/cell_0/gru_cell/gates/kernel/readIdentity/rnn/multi_rnn_cell/cell_0/gru_cell/gates/kernel*
T0
�
Krnn/while/rnn/multi_rnn_cell/cell_0/cell_0/gru_cell/gates/gates/concat/axisConst^rnn/while/Identity*
dtype0*
value	B :
�
Frnn/while/rnn/multi_rnn_cell/cell_0/cell_0/gru_cell/gates/gates/concatConcatV2rnn/while/TensorArrayReadV3rnn/while/Identity_2Krnn/while/rnn/multi_rnn_cell/cell_0/cell_0/gru_cell/gates/gates/concat/axis*

Tidx0*
T0*
N
�
Lrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/gru_cell/gates/gates/MatMul/EnterEnter4rnn/multi_rnn_cell/cell_0/gru_cell/gates/kernel/read*
parallel_iterations *
is_constant(*
T0*$

frame_namernn/while/rnn/while/
�
Frnn/while/rnn/multi_rnn_cell/cell_0/cell_0/gru_cell/gates/gates/MatMulMatMulFrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/gru_cell/gates/gates/concatLrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/gru_cell/gates/gates/MatMul/Enter*
transpose_b( *
transpose_a( *
T0
�
-rnn/multi_rnn_cell/cell_0/gru_cell/gates/biasConst*
dtype0*E
value<B:"0x������?Tl   �?�pS   �?��  �?1{l  �?�֡���?
v
2rnn/multi_rnn_cell/cell_0/gru_cell/gates/bias/readIdentity-rnn/multi_rnn_cell/cell_0/gru_cell/gates/bias*
T0
�
Mrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/gru_cell/gates/gates/BiasAdd/EnterEnter2rnn/multi_rnn_cell/cell_0/gru_cell/gates/bias/read*
parallel_iterations *
is_constant(*
T0*$

frame_namernn/while/rnn/while/
�
Grnn/while/rnn/multi_rnn_cell/cell_0/cell_0/gru_cell/gates/gates/BiasAddBiasAddFrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/gru_cell/gates/gates/MatMulMrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/gru_cell/gates/gates/BiasAdd/Enter*
T0*
data_formatNHWC
�
Arnn/while/rnn/multi_rnn_cell/cell_0/cell_0/gru_cell/gates/SigmoidSigmoidGrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/gru_cell/gates/gates/BiasAdd*
T0
�
Irnn/while/rnn/multi_rnn_cell/cell_0/cell_0/gru_cell/gates/split/split_dimConst^rnn/while/Identity*
dtype0*
value	B :
�
?rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/gru_cell/gates/splitSplitIrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/gru_cell/gates/split/split_dimArnn/while/rnn/multi_rnn_cell/cell_0/cell_0/gru_cell/gates/Sigmoid*
	num_split*
T0
�
Arnn/while/rnn/multi_rnn_cell/cell_0/cell_0/gru_cell/candidate/mulMul?rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/gru_cell/gates/splitrnn/while/Identity_2*
T0
�
3rnn/multi_rnn_cell/cell_0/gru_cell/candidate/kernelConst*
dtype0*�
value�B�	"���,�M�?��\WN��?�����Q�?,�\�:!��||2�z%����Ud�?B����㿏���CX�?�c2�1��?���B��5��Vܿ۶����޿I9B��?C�o�}�?֛�t�?T��9$�R��j�}?���4�4�ӷ'���ʿ0��t޿x�;S�ÿ�Hg$?�¿�Ȥ���˿DD7~h��?�G��C��\ҍ;~ȿOy���?
�
8rnn/multi_rnn_cell/cell_0/gru_cell/candidate/kernel/readIdentity3rnn/multi_rnn_cell/cell_0/gru_cell/candidate/kernel*
T0
�
Srnn/while/rnn/multi_rnn_cell/cell_0/cell_0/gru_cell/candidate/candidate/concat/axisConst^rnn/while/Identity*
dtype0*
value	B :
�
Nrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/gru_cell/candidate/candidate/concatConcatV2rnn/while/TensorArrayReadV3Arnn/while/rnn/multi_rnn_cell/cell_0/cell_0/gru_cell/candidate/mulSrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/gru_cell/candidate/candidate/concat/axis*

Tidx0*
T0*
N
�
Trnn/while/rnn/multi_rnn_cell/cell_0/cell_0/gru_cell/candidate/candidate/MatMul/EnterEnter8rnn/multi_rnn_cell/cell_0/gru_cell/candidate/kernel/read*
parallel_iterations *
is_constant(*
T0*$

frame_namernn/while/rnn/while/
�
Nrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/gru_cell/candidate/candidate/MatMulMatMulNrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/gru_cell/candidate/candidate/concatTrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/gru_cell/candidate/candidate/MatMul/Enter*
transpose_b( *
transpose_a( *
T0
v
1rnn/multi_rnn_cell/cell_0/gru_cell/candidate/biasConst*
dtype0*-
value$B""&���c��>Nm�v����m�3���>
~
6rnn/multi_rnn_cell/cell_0/gru_cell/candidate/bias/readIdentity1rnn/multi_rnn_cell/cell_0/gru_cell/candidate/bias*
T0
�
Urnn/while/rnn/multi_rnn_cell/cell_0/cell_0/gru_cell/candidate/candidate/BiasAdd/EnterEnter6rnn/multi_rnn_cell/cell_0/gru_cell/candidate/bias/read*
parallel_iterations *
is_constant(*
T0*$

frame_namernn/while/rnn/while/
�
Ornn/while/rnn/multi_rnn_cell/cell_0/cell_0/gru_cell/candidate/candidate/BiasAddBiasAddNrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/gru_cell/candidate/candidate/MatMulUrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/gru_cell/candidate/candidate/BiasAdd/Enter*
T0*
data_formatNHWC
�
Brnn/while/rnn/multi_rnn_cell/cell_0/cell_0/gru_cell/candidate/TanhTanhOrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/gru_cell/candidate/candidate/BiasAdd*
T0
�
7rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/gru_cell/mulMulArnn/while/rnn/multi_rnn_cell/cell_0/cell_0/gru_cell/gates/split:1rnn/while/Identity_2*
T0

9rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/gru_cell/sub/xConst^rnn/while/Identity*
dtype0*
valueB 2      �?
�
7rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/gru_cell/subSub9rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/gru_cell/sub/xArnn/while/rnn/multi_rnn_cell/cell_0/cell_0/gru_cell/gates/split:1*
T0
�
9rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/gru_cell/mul_1Mul7rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/gru_cell/subBrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/gru_cell/candidate/Tanh*
T0
�
7rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/gru_cell/addAdd7rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/gru_cell/mul9rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/gru_cell/mul_1*
T0
�
/rnn/multi_rnn_cell/cell_1/gru_cell/gates/kernelConst*
dtype0*�
value�B�"�����eA�?L/�W;�?��+�����+��B�6�6v���?/~�����?��a��O�?�F1x�W�?��C��ؿ�c��ÓԿ d�ќ�?.^�%��?�Q���?=�y�<߿Ӹ�{Ro�?&�G��?��o#c�?���LTJ��W��c��?^�- �Ŀ�WV᷿�ׯ�4��?c��?�6���ҿU��6����"�|��r�q�Π����a"����Cs���?�����	X��~�W��׿��*����?���@P��s��o��?>��xA�?J� ��P�?
z
4rnn/multi_rnn_cell/cell_1/gru_cell/gates/kernel/readIdentity/rnn/multi_rnn_cell/cell_1/gru_cell/gates/kernel*
T0
�
Krnn/while/rnn/multi_rnn_cell/cell_1/cell_1/gru_cell/gates/gates/concat/axisConst^rnn/while/Identity*
dtype0*
value	B :
�
Frnn/while/rnn/multi_rnn_cell/cell_1/cell_1/gru_cell/gates/gates/concatConcatV27rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/gru_cell/addrnn/while/Identity_3Krnn/while/rnn/multi_rnn_cell/cell_1/cell_1/gru_cell/gates/gates/concat/axis*

Tidx0*
T0*
N
�
Lrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/gru_cell/gates/gates/MatMul/EnterEnter4rnn/multi_rnn_cell/cell_1/gru_cell/gates/kernel/read*
parallel_iterations *
is_constant(*
T0*$

frame_namernn/while/rnn/while/
�
Frnn/while/rnn/multi_rnn_cell/cell_1/cell_1/gru_cell/gates/gates/MatMulMatMulFrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/gru_cell/gates/gates/concatLrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/gru_cell/gates/gates/MatMul/Enter*
transpose_b( *
transpose_a( *
T0
�
-rnn/multi_rnn_cell/cell_1/gru_cell/gates/biasConst*
dtype0*E
value<B:"0Nf   �?��   �?������?��y  �?��¦���?ԻK����?
v
2rnn/multi_rnn_cell/cell_1/gru_cell/gates/bias/readIdentity-rnn/multi_rnn_cell/cell_1/gru_cell/gates/bias*
T0
�
Mrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/gru_cell/gates/gates/BiasAdd/EnterEnter2rnn/multi_rnn_cell/cell_1/gru_cell/gates/bias/read*
parallel_iterations *
is_constant(*
T0*$

frame_namernn/while/rnn/while/
�
Grnn/while/rnn/multi_rnn_cell/cell_1/cell_1/gru_cell/gates/gates/BiasAddBiasAddFrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/gru_cell/gates/gates/MatMulMrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/gru_cell/gates/gates/BiasAdd/Enter*
T0*
data_formatNHWC
�
Arnn/while/rnn/multi_rnn_cell/cell_1/cell_1/gru_cell/gates/SigmoidSigmoidGrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/gru_cell/gates/gates/BiasAdd*
T0
�
Irnn/while/rnn/multi_rnn_cell/cell_1/cell_1/gru_cell/gates/split/split_dimConst^rnn/while/Identity*
dtype0*
value	B :
�
?rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/gru_cell/gates/splitSplitIrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/gru_cell/gates/split/split_dimArnn/while/rnn/multi_rnn_cell/cell_1/cell_1/gru_cell/gates/Sigmoid*
	num_split*
T0
�
Arnn/while/rnn/multi_rnn_cell/cell_1/cell_1/gru_cell/candidate/mulMul?rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/gru_cell/gates/splitrnn/while/Identity_3*
T0
�
3rnn/multi_rnn_cell/cell_1/gru_cell/candidate/kernelConst*
dtype0*�
value�B�"�.{0���6�h�yο���Q��߿�^�X���?�	0 �?;k������aO�ۿ@\��.q�?�g�ú��,!��#ܿ��� �ٿ�lW��տ�m�T����GCo,��W3~6��?{?����տɊ��J_�_d���ؿ
�
8rnn/multi_rnn_cell/cell_1/gru_cell/candidate/kernel/readIdentity3rnn/multi_rnn_cell/cell_1/gru_cell/candidate/kernel*
T0
�
Srnn/while/rnn/multi_rnn_cell/cell_1/cell_1/gru_cell/candidate/candidate/concat/axisConst^rnn/while/Identity*
dtype0*
value	B :
�
Nrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/gru_cell/candidate/candidate/concatConcatV27rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/gru_cell/addArnn/while/rnn/multi_rnn_cell/cell_1/cell_1/gru_cell/candidate/mulSrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/gru_cell/candidate/candidate/concat/axis*

Tidx0*
T0*
N
�
Trnn/while/rnn/multi_rnn_cell/cell_1/cell_1/gru_cell/candidate/candidate/MatMul/EnterEnter8rnn/multi_rnn_cell/cell_1/gru_cell/candidate/kernel/read*
parallel_iterations *
is_constant(*
T0*$

frame_namernn/while/rnn/while/
�
Nrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/gru_cell/candidate/candidate/MatMulMatMulNrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/gru_cell/candidate/candidate/concatTrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/gru_cell/candidate/candidate/MatMul/Enter*
transpose_b( *
transpose_a( *
T0
v
1rnn/multi_rnn_cell/cell_1/gru_cell/candidate/biasConst*
dtype0*-
value$B""���پ�K�QD��>�B��B[�>
~
6rnn/multi_rnn_cell/cell_1/gru_cell/candidate/bias/readIdentity1rnn/multi_rnn_cell/cell_1/gru_cell/candidate/bias*
T0
�
Urnn/while/rnn/multi_rnn_cell/cell_1/cell_1/gru_cell/candidate/candidate/BiasAdd/EnterEnter6rnn/multi_rnn_cell/cell_1/gru_cell/candidate/bias/read*
parallel_iterations *
is_constant(*
T0*$

frame_namernn/while/rnn/while/
�
Ornn/while/rnn/multi_rnn_cell/cell_1/cell_1/gru_cell/candidate/candidate/BiasAddBiasAddNrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/gru_cell/candidate/candidate/MatMulUrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/gru_cell/candidate/candidate/BiasAdd/Enter*
T0*
data_formatNHWC
�
Brnn/while/rnn/multi_rnn_cell/cell_1/cell_1/gru_cell/candidate/TanhTanhOrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/gru_cell/candidate/candidate/BiasAdd*
T0
�
7rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/gru_cell/mulMulArnn/while/rnn/multi_rnn_cell/cell_1/cell_1/gru_cell/gates/split:1rnn/while/Identity_3*
T0

9rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/gru_cell/sub/xConst^rnn/while/Identity*
dtype0*
valueB 2      �?
�
7rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/gru_cell/subSub9rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/gru_cell/sub/xArnn/while/rnn/multi_rnn_cell/cell_1/cell_1/gru_cell/gates/split:1*
T0
�
9rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/gru_cell/mul_1Mul7rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/gru_cell/subBrnn/while/rnn/multi_rnn_cell/cell_1/cell_1/gru_cell/candidate/Tanh*
T0
�
7rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/gru_cell/addAdd7rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/gru_cell/mul9rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/gru_cell/mul_1*
T0
�
/rnn/multi_rnn_cell/cell_2/gru_cell/gates/kernelConst*
dtype0*�
value�B�"��l���ҿ���P�Iп]H�<A�?ㄨ\Eǿf^�UF(ݿ��/��/��G-�ڀǟ�� ���rѿ=up�K"�?�)J�j��ͧ)|.�~`�ux]Կ,H�}�]�?l�"WT?�+�4����?�vY���̿�MEcb�?&�� 䠿O��
vĿ�P8���?N=����ο�/��A�>�;W��?tn�o��?��R�e濌f�I���aM�^�]ۿ��d��ƿBv�h�?��-)��?�a��o�?�&��>�?w��9=eʿX����˿�Kn����D`P:�?
z
4rnn/multi_rnn_cell/cell_2/gru_cell/gates/kernel/readIdentity/rnn/multi_rnn_cell/cell_2/gru_cell/gates/kernel*
T0
�
Krnn/while/rnn/multi_rnn_cell/cell_2/cell_2/gru_cell/gates/gates/concat/axisConst^rnn/while/Identity*
dtype0*
value	B :
�
Frnn/while/rnn/multi_rnn_cell/cell_2/cell_2/gru_cell/gates/gates/concatConcatV27rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/gru_cell/addrnn/while/Identity_4Krnn/while/rnn/multi_rnn_cell/cell_2/cell_2/gru_cell/gates/gates/concat/axis*

Tidx0*
T0*
N
�
Lrnn/while/rnn/multi_rnn_cell/cell_2/cell_2/gru_cell/gates/gates/MatMul/EnterEnter4rnn/multi_rnn_cell/cell_2/gru_cell/gates/kernel/read*
parallel_iterations *
is_constant(*
T0*$

frame_namernn/while/rnn/while/
�
Frnn/while/rnn/multi_rnn_cell/cell_2/cell_2/gru_cell/gates/gates/MatMulMatMulFrnn/while/rnn/multi_rnn_cell/cell_2/cell_2/gru_cell/gates/gates/concatLrnn/while/rnn/multi_rnn_cell/cell_2/cell_2/gru_cell/gates/gates/MatMul/Enter*
transpose_b( *
transpose_a( *
T0
�
-rnn/multi_rnn_cell/cell_2/gru_cell/gates/biasConst*
dtype0*E
value<B:"0�R   �?���   �?�g^   �?���   �?��$����?�k�  �?
v
2rnn/multi_rnn_cell/cell_2/gru_cell/gates/bias/readIdentity-rnn/multi_rnn_cell/cell_2/gru_cell/gates/bias*
T0
�
Mrnn/while/rnn/multi_rnn_cell/cell_2/cell_2/gru_cell/gates/gates/BiasAdd/EnterEnter2rnn/multi_rnn_cell/cell_2/gru_cell/gates/bias/read*
parallel_iterations *
is_constant(*
T0*$

frame_namernn/while/rnn/while/
�
Grnn/while/rnn/multi_rnn_cell/cell_2/cell_2/gru_cell/gates/gates/BiasAddBiasAddFrnn/while/rnn/multi_rnn_cell/cell_2/cell_2/gru_cell/gates/gates/MatMulMrnn/while/rnn/multi_rnn_cell/cell_2/cell_2/gru_cell/gates/gates/BiasAdd/Enter*
T0*
data_formatNHWC
�
Arnn/while/rnn/multi_rnn_cell/cell_2/cell_2/gru_cell/gates/SigmoidSigmoidGrnn/while/rnn/multi_rnn_cell/cell_2/cell_2/gru_cell/gates/gates/BiasAdd*
T0
�
Irnn/while/rnn/multi_rnn_cell/cell_2/cell_2/gru_cell/gates/split/split_dimConst^rnn/while/Identity*
dtype0*
value	B :
�
?rnn/while/rnn/multi_rnn_cell/cell_2/cell_2/gru_cell/gates/splitSplitIrnn/while/rnn/multi_rnn_cell/cell_2/cell_2/gru_cell/gates/split/split_dimArnn/while/rnn/multi_rnn_cell/cell_2/cell_2/gru_cell/gates/Sigmoid*
	num_split*
T0
�
Arnn/while/rnn/multi_rnn_cell/cell_2/cell_2/gru_cell/candidate/mulMul?rnn/while/rnn/multi_rnn_cell/cell_2/cell_2/gru_cell/gates/splitrnn/while/Identity_4*
T0
�
3rnn/multi_rnn_cell/cell_2/gru_cell/candidate/kernelConst*
dtype0*�
value�B�"�a�}u���{��G��ٺ����^ͯa�[���I"c�?�����ܿ=�%z�ۿ	�K�}n�?��\D��ҿl�l3Ew�?��?O�_�'�̈́�P�?��"A_
�?�����?B�.���ѿ��+@��?�F"�y�ӿğ�@�_��
�
8rnn/multi_rnn_cell/cell_2/gru_cell/candidate/kernel/readIdentity3rnn/multi_rnn_cell/cell_2/gru_cell/candidate/kernel*
T0
�
Srnn/while/rnn/multi_rnn_cell/cell_2/cell_2/gru_cell/candidate/candidate/concat/axisConst^rnn/while/Identity*
dtype0*
value	B :
�
Nrnn/while/rnn/multi_rnn_cell/cell_2/cell_2/gru_cell/candidate/candidate/concatConcatV27rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/gru_cell/addArnn/while/rnn/multi_rnn_cell/cell_2/cell_2/gru_cell/candidate/mulSrnn/while/rnn/multi_rnn_cell/cell_2/cell_2/gru_cell/candidate/candidate/concat/axis*

Tidx0*
T0*
N
�
Trnn/while/rnn/multi_rnn_cell/cell_2/cell_2/gru_cell/candidate/candidate/MatMul/EnterEnter8rnn/multi_rnn_cell/cell_2/gru_cell/candidate/kernel/read*
parallel_iterations *
is_constant(*
T0*$

frame_namernn/while/rnn/while/
�
Nrnn/while/rnn/multi_rnn_cell/cell_2/cell_2/gru_cell/candidate/candidate/MatMulMatMulNrnn/while/rnn/multi_rnn_cell/cell_2/cell_2/gru_cell/candidate/candidate/concatTrnn/while/rnn/multi_rnn_cell/cell_2/cell_2/gru_cell/candidate/candidate/MatMul/Enter*
transpose_b( *
transpose_a( *
T0
v
1rnn/multi_rnn_cell/cell_2/gru_cell/candidate/biasConst*
dtype0*-
value$B""�c�{#��>}��J�L�>D������>
~
6rnn/multi_rnn_cell/cell_2/gru_cell/candidate/bias/readIdentity1rnn/multi_rnn_cell/cell_2/gru_cell/candidate/bias*
T0
�
Urnn/while/rnn/multi_rnn_cell/cell_2/cell_2/gru_cell/candidate/candidate/BiasAdd/EnterEnter6rnn/multi_rnn_cell/cell_2/gru_cell/candidate/bias/read*
parallel_iterations *
is_constant(*
T0*$

frame_namernn/while/rnn/while/
�
Ornn/while/rnn/multi_rnn_cell/cell_2/cell_2/gru_cell/candidate/candidate/BiasAddBiasAddNrnn/while/rnn/multi_rnn_cell/cell_2/cell_2/gru_cell/candidate/candidate/MatMulUrnn/while/rnn/multi_rnn_cell/cell_2/cell_2/gru_cell/candidate/candidate/BiasAdd/Enter*
T0*
data_formatNHWC
�
Brnn/while/rnn/multi_rnn_cell/cell_2/cell_2/gru_cell/candidate/TanhTanhOrnn/while/rnn/multi_rnn_cell/cell_2/cell_2/gru_cell/candidate/candidate/BiasAdd*
T0
�
7rnn/while/rnn/multi_rnn_cell/cell_2/cell_2/gru_cell/mulMulArnn/while/rnn/multi_rnn_cell/cell_2/cell_2/gru_cell/gates/split:1rnn/while/Identity_4*
T0

9rnn/while/rnn/multi_rnn_cell/cell_2/cell_2/gru_cell/sub/xConst^rnn/while/Identity*
dtype0*
valueB 2      �?
�
7rnn/while/rnn/multi_rnn_cell/cell_2/cell_2/gru_cell/subSub9rnn/while/rnn/multi_rnn_cell/cell_2/cell_2/gru_cell/sub/xArnn/while/rnn/multi_rnn_cell/cell_2/cell_2/gru_cell/gates/split:1*
T0
�
9rnn/while/rnn/multi_rnn_cell/cell_2/cell_2/gru_cell/mul_1Mul7rnn/while/rnn/multi_rnn_cell/cell_2/cell_2/gru_cell/subBrnn/while/rnn/multi_rnn_cell/cell_2/cell_2/gru_cell/candidate/Tanh*
T0
�
7rnn/while/rnn/multi_rnn_cell/cell_2/cell_2/gru_cell/addAdd7rnn/while/rnn/multi_rnn_cell/cell_2/cell_2/gru_cell/mul9rnn/while/rnn/multi_rnn_cell/cell_2/cell_2/gru_cell/mul_1*
T0
�
/rnn/multi_rnn_cell/cell_3/gru_cell/gates/kernelConst*
dtype0*�
value�B�"���o b$�?�3jJ���wI�����?�ֳ���?'d+�����t�z
M��4���տ%��&A�Q-_�c�ſ �#��s�?L��?�Y ���N �s�.���v�?3�<�qԿ��v��ʿy�&���KM�ѿ	����?U�Dd` ݿϞ��޿��e��?q����?�luUN��?�4���\Կ����ފ��������Rd
���?�?�ֺ��WȵCk?�e�����?"�-;�E�?f�z��?s��:�Iп"�$MR�Ͽ����Qڿ
z
4rnn/multi_rnn_cell/cell_3/gru_cell/gates/kernel/readIdentity/rnn/multi_rnn_cell/cell_3/gru_cell/gates/kernel*
T0
�
Krnn/while/rnn/multi_rnn_cell/cell_3/cell_3/gru_cell/gates/gates/concat/axisConst^rnn/while/Identity*
dtype0*
value	B :
�
Frnn/while/rnn/multi_rnn_cell/cell_3/cell_3/gru_cell/gates/gates/concatConcatV27rnn/while/rnn/multi_rnn_cell/cell_2/cell_2/gru_cell/addrnn/while/Identity_5Krnn/while/rnn/multi_rnn_cell/cell_3/cell_3/gru_cell/gates/gates/concat/axis*

Tidx0*
T0*
N
�
Lrnn/while/rnn/multi_rnn_cell/cell_3/cell_3/gru_cell/gates/gates/MatMul/EnterEnter4rnn/multi_rnn_cell/cell_3/gru_cell/gates/kernel/read*
parallel_iterations *
is_constant(*
T0*$

frame_namernn/while/rnn/while/
�
Frnn/while/rnn/multi_rnn_cell/cell_3/cell_3/gru_cell/gates/gates/MatMulMatMulFrnn/while/rnn/multi_rnn_cell/cell_3/cell_3/gru_cell/gates/gates/concatLrnn/while/rnn/multi_rnn_cell/cell_3/cell_3/gru_cell/gates/gates/MatMul/Enter*
transpose_b( *
transpose_a( *
T0
�
-rnn/multi_rnn_cell/cell_3/gru_cell/gates/biasConst*
dtype0*E
value<B:"0q�M����?0 _   �?�ro  �?�uH���?��'����?-P�r  �?
v
2rnn/multi_rnn_cell/cell_3/gru_cell/gates/bias/readIdentity-rnn/multi_rnn_cell/cell_3/gru_cell/gates/bias*
T0
�
Mrnn/while/rnn/multi_rnn_cell/cell_3/cell_3/gru_cell/gates/gates/BiasAdd/EnterEnter2rnn/multi_rnn_cell/cell_3/gru_cell/gates/bias/read*
parallel_iterations *
is_constant(*
T0*$

frame_namernn/while/rnn/while/
�
Grnn/while/rnn/multi_rnn_cell/cell_3/cell_3/gru_cell/gates/gates/BiasAddBiasAddFrnn/while/rnn/multi_rnn_cell/cell_3/cell_3/gru_cell/gates/gates/MatMulMrnn/while/rnn/multi_rnn_cell/cell_3/cell_3/gru_cell/gates/gates/BiasAdd/Enter*
T0*
data_formatNHWC
�
Arnn/while/rnn/multi_rnn_cell/cell_3/cell_3/gru_cell/gates/SigmoidSigmoidGrnn/while/rnn/multi_rnn_cell/cell_3/cell_3/gru_cell/gates/gates/BiasAdd*
T0
�
Irnn/while/rnn/multi_rnn_cell/cell_3/cell_3/gru_cell/gates/split/split_dimConst^rnn/while/Identity*
dtype0*
value	B :
�
?rnn/while/rnn/multi_rnn_cell/cell_3/cell_3/gru_cell/gates/splitSplitIrnn/while/rnn/multi_rnn_cell/cell_3/cell_3/gru_cell/gates/split/split_dimArnn/while/rnn/multi_rnn_cell/cell_3/cell_3/gru_cell/gates/Sigmoid*
	num_split*
T0
�
Arnn/while/rnn/multi_rnn_cell/cell_3/cell_3/gru_cell/candidate/mulMul?rnn/while/rnn/multi_rnn_cell/cell_3/cell_3/gru_cell/gates/splitrnn/while/Identity_5*
T0
�
3rnn/multi_rnn_cell/cell_3/gru_cell/candidate/kernelConst*
dtype0*�
value�B�"���d9E�ڗp��q�?o>���G�鿬�����?z�N2���G�·?�N��X�ÿ��1�ܿ�?�͉��տ�x����?Dv�~4 ɿ	)� �c��Z����?�/.���wr�Z������Կ���y�r�
�
8rnn/multi_rnn_cell/cell_3/gru_cell/candidate/kernel/readIdentity3rnn/multi_rnn_cell/cell_3/gru_cell/candidate/kernel*
T0
�
Srnn/while/rnn/multi_rnn_cell/cell_3/cell_3/gru_cell/candidate/candidate/concat/axisConst^rnn/while/Identity*
dtype0*
value	B :
�
Nrnn/while/rnn/multi_rnn_cell/cell_3/cell_3/gru_cell/candidate/candidate/concatConcatV27rnn/while/rnn/multi_rnn_cell/cell_2/cell_2/gru_cell/addArnn/while/rnn/multi_rnn_cell/cell_3/cell_3/gru_cell/candidate/mulSrnn/while/rnn/multi_rnn_cell/cell_3/cell_3/gru_cell/candidate/candidate/concat/axis*

Tidx0*
T0*
N
�
Trnn/while/rnn/multi_rnn_cell/cell_3/cell_3/gru_cell/candidate/candidate/MatMul/EnterEnter8rnn/multi_rnn_cell/cell_3/gru_cell/candidate/kernel/read*
parallel_iterations *
is_constant(*
T0*$

frame_namernn/while/rnn/while/
�
Nrnn/while/rnn/multi_rnn_cell/cell_3/cell_3/gru_cell/candidate/candidate/MatMulMatMulNrnn/while/rnn/multi_rnn_cell/cell_3/cell_3/gru_cell/candidate/candidate/concatTrnn/while/rnn/multi_rnn_cell/cell_3/cell_3/gru_cell/candidate/candidate/MatMul/Enter*
transpose_b( *
transpose_a( *
T0
v
1rnn/multi_rnn_cell/cell_3/gru_cell/candidate/biasConst*
dtype0*-
value$B""����wK(�'IUf*?�D$o<+?
~
6rnn/multi_rnn_cell/cell_3/gru_cell/candidate/bias/readIdentity1rnn/multi_rnn_cell/cell_3/gru_cell/candidate/bias*
T0
�
Urnn/while/rnn/multi_rnn_cell/cell_3/cell_3/gru_cell/candidate/candidate/BiasAdd/EnterEnter6rnn/multi_rnn_cell/cell_3/gru_cell/candidate/bias/read*
parallel_iterations *
is_constant(*
T0*$

frame_namernn/while/rnn/while/
�
Ornn/while/rnn/multi_rnn_cell/cell_3/cell_3/gru_cell/candidate/candidate/BiasAddBiasAddNrnn/while/rnn/multi_rnn_cell/cell_3/cell_3/gru_cell/candidate/candidate/MatMulUrnn/while/rnn/multi_rnn_cell/cell_3/cell_3/gru_cell/candidate/candidate/BiasAdd/Enter*
T0*
data_formatNHWC
�
Brnn/while/rnn/multi_rnn_cell/cell_3/cell_3/gru_cell/candidate/TanhTanhOrnn/while/rnn/multi_rnn_cell/cell_3/cell_3/gru_cell/candidate/candidate/BiasAdd*
T0
�
7rnn/while/rnn/multi_rnn_cell/cell_3/cell_3/gru_cell/mulMulArnn/while/rnn/multi_rnn_cell/cell_3/cell_3/gru_cell/gates/split:1rnn/while/Identity_5*
T0

9rnn/while/rnn/multi_rnn_cell/cell_3/cell_3/gru_cell/sub/xConst^rnn/while/Identity*
dtype0*
valueB 2      �?
�
7rnn/while/rnn/multi_rnn_cell/cell_3/cell_3/gru_cell/subSub9rnn/while/rnn/multi_rnn_cell/cell_3/cell_3/gru_cell/sub/xArnn/while/rnn/multi_rnn_cell/cell_3/cell_3/gru_cell/gates/split:1*
T0
�
9rnn/while/rnn/multi_rnn_cell/cell_3/cell_3/gru_cell/mul_1Mul7rnn/while/rnn/multi_rnn_cell/cell_3/cell_3/gru_cell/subBrnn/while/rnn/multi_rnn_cell/cell_3/cell_3/gru_cell/candidate/Tanh*
T0
�
7rnn/while/rnn/multi_rnn_cell/cell_3/cell_3/gru_cell/addAdd7rnn/while/rnn/multi_rnn_cell/cell_3/cell_3/gru_cell/mul9rnn/while/rnn/multi_rnn_cell/cell_3/cell_3/gru_cell/mul_1*
T0
�
/rnn/multi_rnn_cell/cell_4/gru_cell/gates/kernelConst*
dtype0*�
value�B�"�)A}�~l�? A���޿>�~���o6�l��?Q9�_�|�?3�zq��Կ.��`c�?E+s9��?�N��tſfv`)�vܿ�c��ڿ����W���mV��ܿ[�
���?��(�+ÿg�����}��ҿ9Z�=OԿ���)�8ֿ8$�Crÿc�zX��?��Z���?�=}�}�?�Y��h˿ڧ	�V_�?�R-ݿq�%�˻��) ޏ�ؿ���Ϳ�;�w�߿�۱n���6+�ؽ�d��n� ʿ6��D�?���!ZS�?�f"�Z�?
z
4rnn/multi_rnn_cell/cell_4/gru_cell/gates/kernel/readIdentity/rnn/multi_rnn_cell/cell_4/gru_cell/gates/kernel*
T0
�
Krnn/while/rnn/multi_rnn_cell/cell_4/cell_4/gru_cell/gates/gates/concat/axisConst^rnn/while/Identity*
dtype0*
value	B :
�
Frnn/while/rnn/multi_rnn_cell/cell_4/cell_4/gru_cell/gates/gates/concatConcatV27rnn/while/rnn/multi_rnn_cell/cell_3/cell_3/gru_cell/addrnn/while/Identity_6Krnn/while/rnn/multi_rnn_cell/cell_4/cell_4/gru_cell/gates/gates/concat/axis*

Tidx0*
T0*
N
�
Lrnn/while/rnn/multi_rnn_cell/cell_4/cell_4/gru_cell/gates/gates/MatMul/EnterEnter4rnn/multi_rnn_cell/cell_4/gru_cell/gates/kernel/read*
parallel_iterations *
is_constant(*
T0*$

frame_namernn/while/rnn/while/
�
Frnn/while/rnn/multi_rnn_cell/cell_4/cell_4/gru_cell/gates/gates/MatMulMatMulFrnn/while/rnn/multi_rnn_cell/cell_4/cell_4/gru_cell/gates/gates/concatLrnn/while/rnn/multi_rnn_cell/cell_4/cell_4/gru_cell/gates/gates/MatMul/Enter*
transpose_b( *
transpose_a( *
T0
�
-rnn/multi_rnn_cell/cell_4/gru_cell/gates/biasConst*
dtype0*E
value<B:"0��K����?x�`   �?�   �?�������?�������?4]&   �?
v
2rnn/multi_rnn_cell/cell_4/gru_cell/gates/bias/readIdentity-rnn/multi_rnn_cell/cell_4/gru_cell/gates/bias*
T0
�
Mrnn/while/rnn/multi_rnn_cell/cell_4/cell_4/gru_cell/gates/gates/BiasAdd/EnterEnter2rnn/multi_rnn_cell/cell_4/gru_cell/gates/bias/read*
parallel_iterations *
is_constant(*
T0*$

frame_namernn/while/rnn/while/
�
Grnn/while/rnn/multi_rnn_cell/cell_4/cell_4/gru_cell/gates/gates/BiasAddBiasAddFrnn/while/rnn/multi_rnn_cell/cell_4/cell_4/gru_cell/gates/gates/MatMulMrnn/while/rnn/multi_rnn_cell/cell_4/cell_4/gru_cell/gates/gates/BiasAdd/Enter*
T0*
data_formatNHWC
�
Arnn/while/rnn/multi_rnn_cell/cell_4/cell_4/gru_cell/gates/SigmoidSigmoidGrnn/while/rnn/multi_rnn_cell/cell_4/cell_4/gru_cell/gates/gates/BiasAdd*
T0
�
Irnn/while/rnn/multi_rnn_cell/cell_4/cell_4/gru_cell/gates/split/split_dimConst^rnn/while/Identity*
dtype0*
value	B :
�
?rnn/while/rnn/multi_rnn_cell/cell_4/cell_4/gru_cell/gates/splitSplitIrnn/while/rnn/multi_rnn_cell/cell_4/cell_4/gru_cell/gates/split/split_dimArnn/while/rnn/multi_rnn_cell/cell_4/cell_4/gru_cell/gates/Sigmoid*
	num_split*
T0
�
Arnn/while/rnn/multi_rnn_cell/cell_4/cell_4/gru_cell/candidate/mulMul?rnn/while/rnn/multi_rnn_cell/cell_4/cell_4/gru_cell/gates/splitrnn/while/Identity_6*
T0
�
3rnn/multi_rnn_cell/cell_4/gru_cell/candidate/kernelConst*
dtype0*�
value�B�"��x�����pN�/5�?���0 ÿW�G;²��up��N������?	jU"�տ�$�ak��G��1��?�@Em�⿺��60�?Xb0~nK�?�ԭ�迳�E�d�?R��)1�?/^2����?��t��0��|�"�
�
8rnn/multi_rnn_cell/cell_4/gru_cell/candidate/kernel/readIdentity3rnn/multi_rnn_cell/cell_4/gru_cell/candidate/kernel*
T0
�
Srnn/while/rnn/multi_rnn_cell/cell_4/cell_4/gru_cell/candidate/candidate/concat/axisConst^rnn/while/Identity*
dtype0*
value	B :
�
Nrnn/while/rnn/multi_rnn_cell/cell_4/cell_4/gru_cell/candidate/candidate/concatConcatV27rnn/while/rnn/multi_rnn_cell/cell_3/cell_3/gru_cell/addArnn/while/rnn/multi_rnn_cell/cell_4/cell_4/gru_cell/candidate/mulSrnn/while/rnn/multi_rnn_cell/cell_4/cell_4/gru_cell/candidate/candidate/concat/axis*

Tidx0*
T0*
N
�
Trnn/while/rnn/multi_rnn_cell/cell_4/cell_4/gru_cell/candidate/candidate/MatMul/EnterEnter8rnn/multi_rnn_cell/cell_4/gru_cell/candidate/kernel/read*
parallel_iterations *
is_constant(*
T0*$

frame_namernn/while/rnn/while/
�
Nrnn/while/rnn/multi_rnn_cell/cell_4/cell_4/gru_cell/candidate/candidate/MatMulMatMulNrnn/while/rnn/multi_rnn_cell/cell_4/cell_4/gru_cell/candidate/candidate/concatTrnn/while/rnn/multi_rnn_cell/cell_4/cell_4/gru_cell/candidate/candidate/MatMul/Enter*
transpose_b( *
transpose_a( *
T0
v
1rnn/multi_rnn_cell/cell_4/gru_cell/candidate/biasConst*
dtype0*-
value$B""06�R�{$?�zh:6�=��Գ-%�@?
~
6rnn/multi_rnn_cell/cell_4/gru_cell/candidate/bias/readIdentity1rnn/multi_rnn_cell/cell_4/gru_cell/candidate/bias*
T0
�
Urnn/while/rnn/multi_rnn_cell/cell_4/cell_4/gru_cell/candidate/candidate/BiasAdd/EnterEnter6rnn/multi_rnn_cell/cell_4/gru_cell/candidate/bias/read*
parallel_iterations *
is_constant(*
T0*$

frame_namernn/while/rnn/while/
�
Ornn/while/rnn/multi_rnn_cell/cell_4/cell_4/gru_cell/candidate/candidate/BiasAddBiasAddNrnn/while/rnn/multi_rnn_cell/cell_4/cell_4/gru_cell/candidate/candidate/MatMulUrnn/while/rnn/multi_rnn_cell/cell_4/cell_4/gru_cell/candidate/candidate/BiasAdd/Enter*
T0*
data_formatNHWC
�
Brnn/while/rnn/multi_rnn_cell/cell_4/cell_4/gru_cell/candidate/TanhTanhOrnn/while/rnn/multi_rnn_cell/cell_4/cell_4/gru_cell/candidate/candidate/BiasAdd*
T0
�
7rnn/while/rnn/multi_rnn_cell/cell_4/cell_4/gru_cell/mulMulArnn/while/rnn/multi_rnn_cell/cell_4/cell_4/gru_cell/gates/split:1rnn/while/Identity_6*
T0

9rnn/while/rnn/multi_rnn_cell/cell_4/cell_4/gru_cell/sub/xConst^rnn/while/Identity*
dtype0*
valueB 2      �?
�
7rnn/while/rnn/multi_rnn_cell/cell_4/cell_4/gru_cell/subSub9rnn/while/rnn/multi_rnn_cell/cell_4/cell_4/gru_cell/sub/xArnn/while/rnn/multi_rnn_cell/cell_4/cell_4/gru_cell/gates/split:1*
T0
�
9rnn/while/rnn/multi_rnn_cell/cell_4/cell_4/gru_cell/mul_1Mul7rnn/while/rnn/multi_rnn_cell/cell_4/cell_4/gru_cell/subBrnn/while/rnn/multi_rnn_cell/cell_4/cell_4/gru_cell/candidate/Tanh*
T0
�
7rnn/while/rnn/multi_rnn_cell/cell_4/cell_4/gru_cell/addAdd7rnn/while/rnn/multi_rnn_cell/cell_4/cell_4/gru_cell/mul9rnn/while/rnn/multi_rnn_cell/cell_4/cell_4/gru_cell/mul_1*
T0
�
3rnn/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnterrnn/TensorArray*
parallel_iterations *J
_class@
><loc:@rnn/while/rnn/multi_rnn_cell/cell_4/cell_4/gru_cell/add*
is_constant(*
T0*$

frame_namernn/while/rnn/while/
�
-rnn/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV33rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enterrnn/while/Identity7rnn/while/rnn/multi_rnn_cell/cell_4/cell_4/gru_cell/addrnn/while/Identity_1*J
_class@
><loc:@rnn/while/rnn/multi_rnn_cell/cell_4/cell_4/gru_cell/add*
T0
N
rnn/while/add/yConst^rnn/while/Identity*
dtype0*
value	B :
B
rnn/while/addAddrnn/while/Identityrnn/while/add/y*
T0
@
rnn/while/NextIterationNextIterationrnn/while/add*
T0
b
rnn/while/NextIteration_1NextIteration-rnn/while/TensorArrayWrite/TensorArrayWriteV3*
T0
l
rnn/while/NextIteration_2NextIteration7rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/gru_cell/add*
T0
l
rnn/while/NextIteration_3NextIteration7rnn/while/rnn/multi_rnn_cell/cell_1/cell_1/gru_cell/add*
T0
l
rnn/while/NextIteration_4NextIteration7rnn/while/rnn/multi_rnn_cell/cell_2/cell_2/gru_cell/add*
T0
l
rnn/while/NextIteration_5NextIteration7rnn/while/rnn/multi_rnn_cell/cell_3/cell_3/gru_cell/add*
T0
l
rnn/while/NextIteration_6NextIteration7rnn/while/rnn/multi_rnn_cell/cell_4/cell_4/gru_cell/add*
T0
5
rnn/while/Exit_1Exitrnn/while/Switch_1*
T0
�
&rnn/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3rnn/TensorArrayrnn/while/Exit_1*"
_class
loc:@rnn/TensorArray
n
 rnn/TensorArrayStack/range/startConst*
dtype0*"
_class
loc:@rnn/TensorArray*
value	B : 
n
 rnn/TensorArrayStack/range/deltaConst*
dtype0*"
_class
loc:@rnn/TensorArray*
value	B :
�
rnn/TensorArrayStack/rangeRange rnn/TensorArrayStack/range/start&rnn/TensorArrayStack/TensorArraySizeV3 rnn/TensorArrayStack/range/delta*"
_class
loc:@rnn/TensorArray*

Tidx0
�
(rnn/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3rnn/TensorArrayrnn/TensorArrayStack/rangernn/while/Exit_1*
dtype0*"
_class
loc:@rnn/TensorArray*$
element_shape:���������
2
rnn/RankConst*
dtype0*
value	B :
9
rnn/range/startConst*
dtype0*
value	B :
9
rnn/range/deltaConst*
dtype0*
value	B :
J
	rnn/rangeRangernn/range/startrnn/Rankrnn/range/delta*

Tidx0
J
rnn/concat_1/values_0Const*
dtype0*
valueB"       
;
rnn/concat_1/axisConst*
dtype0*
value	B : 
k
rnn/concat_1ConcatV2rnn/concat_1/values_0	rnn/rangernn/concat_1/axis*

Tidx0*
T0*
N
h
rnn/transpose	Transpose(rnn/TensorArrayStack/TensorArrayGatherV3rnn/concat_1*
Tperm0*
T0
I
transpose_1/permConst*
dtype0*!
valueB"          
O
transpose_1	Transposernn/transposetranspose_1/perm*
Tperm0*
T0
8
Gather/indicesConst*
dtype0*
value	B :
d
GatherGathertranspose_1Gather/indices*
validate_indices(*
Tparams0*
Tindices0
k

Variable_2Const*
dtype0*I
value@B>"0u<&�?�� Z5j?+
�߅��f��'�s?I��d�u?˗��u��
O
Variable_2/readIdentity
Variable_2*
_class
loc:@Variable_2*
T0
G

Variable_3Const*
dtype0*%
valueB"��P���?���#m��?
O
Variable_3/readIdentity
Variable_3*
_class
loc:@Variable_3*
T0
X
MatMulMatMulGatherVariable_2/read*
transpose_b( *
transpose_a( *
T0
,
addAddMatMulVariable_3/read*
T0

outputSoftmaxadd*
T0 