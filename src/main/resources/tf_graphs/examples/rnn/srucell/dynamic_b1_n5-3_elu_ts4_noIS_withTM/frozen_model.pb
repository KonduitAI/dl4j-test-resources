
�
in_0Const*m
valuedBb"P~^G?LM?�p9?�ol>�%:?X�8><q?b|d?��?�al?P@�=,5K?ֹ(?�6?`u#>0�>�{>�h�>�o~?v|?*
dtype0
=
	in_0/readIdentityin_0*
T0*
_class
	loc:@in_0
H
rnn/SRUCellZeroState/ConstConst*
valueB:*
dtype0
J
rnn/SRUCellZeroState/Const_1Const*
valueB:*
dtype0
J
 rnn/SRUCellZeroState/concat/axisConst*
value	B : *
dtype0
�
rnn/SRUCellZeroState/concatConcatV2rnn/SRUCellZeroState/Constrnn/SRUCellZeroState/Const_1 rnn/SRUCellZeroState/concat/axis*
T0*
N*

Tidx0
M
 rnn/SRUCellZeroState/zeros/ConstConst*
valueB
 *    *
dtype0
|
rnn/SRUCellZeroState/zerosFillrnn/SRUCellZeroState/concat rnn/SRUCellZeroState/zeros/Const*
T0*

index_type0
B
	rnn/ShapeConst*!
valueB"         *
dtype0
E
rnn/strided_slice/stackConst*
valueB: *
dtype0
G
rnn/strided_slice/stack_1Const*
valueB:*
dtype0
G
rnn/strided_slice/stack_2Const*
valueB:*
dtype0
�
rnn/strided_sliceStridedSlice	rnn/Shapernn/strided_slice/stackrnn/strided_slice/stack_1rnn/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask 
2
rnn/timeConst*
dtype0*
value	B : 
�
rnn/TensorArrayTensorArrayV3rnn/strided_slice*/
tensor_array_namernn/dynamic_rnn/output_0*
dtype0*
element_shape
:*
clear_after_read(*
dynamic_size( *
identical_element_shapes(
�
rnn/TensorArray_1TensorArrayV3rnn/strided_slice*
element_shape
:*
clear_after_read(*
dynamic_size( *
identical_element_shapes(*.
tensor_array_namernn/dynamic_rnn/input_0*
dtype0
U
rnn/TensorArrayUnstack/ShapeConst*!
valueB"         *
dtype0
X
*rnn/TensorArrayUnstack/strided_slice/stackConst*
dtype0*
valueB: 
Z
,rnn/TensorArrayUnstack/strided_slice/stack_1Const*
valueB:*
dtype0
Z
,rnn/TensorArrayUnstack/strided_slice/stack_2Const*
valueB:*
dtype0
�
$rnn/TensorArrayUnstack/strided_sliceStridedSlicernn/TensorArrayUnstack/Shape*rnn/TensorArrayUnstack/strided_slice/stack,rnn/TensorArrayUnstack/strided_slice/stack_1,rnn/TensorArrayUnstack/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask 
L
"rnn/TensorArrayUnstack/range/startConst*
value	B : *
dtype0
L
"rnn/TensorArrayUnstack/range/deltaConst*
dtype0*
value	B :
�
rnn/TensorArrayUnstack/rangeRange"rnn/TensorArrayUnstack/range/start$rnn/TensorArrayUnstack/strided_slice"rnn/TensorArrayUnstack/range/delta*

Tidx0
�
>rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3rnn/TensorArray_1rnn/TensorArrayUnstack/range	in_0/readrnn/TensorArray_1:1*
T0*
_class
	loc:@in_0
7
rnn/Maximum/xConst*
dtype0*
value	B :
A
rnn/MaximumMaximumrnn/Maximum/xrnn/strided_slice*
T0
?
rnn/MinimumMinimumrnn/strided_slicernn/Maximum*
T0
E
rnn/while/iteration_counterConst*
dtype0*
value	B : 
�
rnn/while/EnterEnterrnn/while/iteration_counter*
T0*
is_constant( *
parallel_iterations *'

frame_namernn/while/while_context
�
rnn/while/Enter_1Enterrnn/time*
T0*
is_constant( *
parallel_iterations *'

frame_namernn/while/while_context
�
rnn/while/Enter_2Enterrnn/TensorArray:1*'

frame_namernn/while/while_context*
T0*
is_constant( *
parallel_iterations 
�
rnn/while/Enter_3Enterrnn/SRUCellZeroState/zeros*
is_constant( *
parallel_iterations *'

frame_namernn/while/while_context*
T0
T
rnn/while/MergeMergernn/while/Enterrnn/while/NextIteration*
N*
T0
Z
rnn/while/Merge_1Mergernn/while/Enter_1rnn/while/NextIteration_1*
T0*
N
Z
rnn/while/Merge_2Mergernn/while/Enter_2rnn/while/NextIteration_2*
N*
T0
Z
rnn/while/Merge_3Mergernn/while/Enter_3rnn/while/NextIteration_3*
N*
T0
F
rnn/while/LessLessrnn/while/Mergernn/while/Less/Enter*
T0
�
rnn/while/Less/EnterEnterrnn/strided_slice*'

frame_namernn/while/while_context*
T0*
is_constant(*
parallel_iterations 
L
rnn/while/Less_1Lessrnn/while/Merge_1rnn/while/Less_1/Enter*
T0
�
rnn/while/Less_1/EnterEnterrnn/Minimum*
T0*
is_constant(*
parallel_iterations *'

frame_namernn/while/while_context
D
rnn/while/LogicalAnd
LogicalAndrnn/while/Lessrnn/while/Less_1
4
rnn/while/LoopCondLoopCondrnn/while/LogicalAnd
l
rnn/while/SwitchSwitchrnn/while/Mergernn/while/LoopCond*
T0*"
_class
loc:@rnn/while/Merge
r
rnn/while/Switch_1Switchrnn/while/Merge_1rnn/while/LoopCond*
T0*$
_class
loc:@rnn/while/Merge_1
r
rnn/while/Switch_2Switchrnn/while/Merge_2rnn/while/LoopCond*
T0*$
_class
loc:@rnn/while/Merge_2
r
rnn/while/Switch_3Switchrnn/while/Merge_3rnn/while/LoopCond*
T0*$
_class
loc:@rnn/while/Merge_3
;
rnn/while/IdentityIdentityrnn/while/Switch:1*
T0
?
rnn/while/Identity_1Identityrnn/while/Switch_1:1*
T0
?
rnn/while/Identity_2Identityrnn/while/Switch_2:1*
T0
?
rnn/while/Identity_3Identityrnn/while/Switch_3:1*
T0
N
rnn/while/add/yConst^rnn/while/Identity*
value	B :*
dtype0
B
rnn/while/addAddrnn/while/Identityrnn/while/add/y*
T0
�
rnn/while/TensorArrayReadV3TensorArrayReadV3!rnn/while/TensorArrayReadV3/Enterrnn/while/Identity_1#rnn/while/TensorArrayReadV3/Enter_1*
dtype0
�
!rnn/while/TensorArrayReadV3/EnterEnterrnn/TensorArray_1*
is_constant(*
parallel_iterations *'

frame_namernn/while/while_context*
T0
�
#rnn/while/TensorArrayReadV3/Enter_1Enter>rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*'

frame_namernn/while/while_context*
T0*
is_constant(*
parallel_iterations 
�
rnn/sru_cell/kernelConst*�
value�B�"�*�z�z�?Bw�>�Q�>��� ��>���=�lJ>�3>@��=���4�� �%:���>p���,ξ�=�H��5�<�����!��`	?m��������?���>���t�ϾI�
��?�O�>�dg�(뛾@���Qa>0��>l��>���>�#%� W��?+����,�<�����?n�?�`ξ�����Q��^����ƾ�x����ͼ����? _Ծ�s
>��G>�H����~�*
dtype0
B
rnn/sru_cell/kernel/readIdentityrnn/sru_cell/kernel*
T0
V
rnn/sru_cell/biasConst*-
value$B""                        *
dtype0
>
rnn/sru_cell/bias/readIdentityrnn/sru_cell/bias*
T0
�
rnn/while/sru_cell/MatMulMatMulrnn/while/TensorArrayReadV3rnn/while/sru_cell/MatMul/Enter*
T0*
transpose_a( *
transpose_b( 
�
rnn/while/sru_cell/MatMul/EnterEnterrnn/sru_cell/kernel/read*
T0*
is_constant(*
parallel_iterations *'

frame_namernn/while/while_context
a
"rnn/while/sru_cell/split/split_dimConst^rnn/while/Identity*
value	B :*
dtype0
z
rnn/while/sru_cell/splitSplit"rnn/while/sru_cell/split/split_dimrnn/while/sru_cell/MatMul*
T0*
	num_split
]
rnn/while/sru_cell/concat/axisConst^rnn/while/Identity*
value	B :*
dtype0
�
rnn/while/sru_cell/concatConcatV2rnn/while/sru_cell/split:1rnn/while/sru_cell/split:2rnn/while/sru_cell/concat/axis*
N*

Tidx0*
T0
�
rnn/while/sru_cell/BiasAddBiasAddrnn/while/sru_cell/concat rnn/while/sru_cell/BiasAdd/Enter*
T0*
data_formatNHWC
�
 rnn/while/sru_cell/BiasAdd/EnterEnterrnn/sru_cell/bias/read*
T0*
is_constant(*
parallel_iterations *'

frame_namernn/while/while_context
J
rnn/while/sru_cell/SigmoidSigmoidrnn/while/sru_cell/BiasAdd*
T0
c
$rnn/while/sru_cell/split_1/split_dimConst^rnn/while/Identity*
value	B :*
dtype0

rnn/while/sru_cell/split_1Split$rnn/while/sru_cell/split_1/split_dimrnn/while/sru_cell/Sigmoid*
T0*
	num_split
X
rnn/while/sru_cell/mulMulrnn/while/sru_cell/split_1rnn/while/Identity_3*
T0
Z
rnn/while/sru_cell/sub/xConst^rnn/while/Identity*
valueB
 *  �?*
dtype0
\
rnn/while/sru_cell/subSubrnn/while/sru_cell/sub/xrnn/while/sru_cell/split_1*
T0
Z
rnn/while/sru_cell/mul_1Mulrnn/while/sru_cell/subrnn/while/sru_cell/split*
T0
X
rnn/while/sru_cell/addAddrnn/while/sru_cell/mulrnn/while/sru_cell/mul_1*
T0
>
rnn/while/sru_cell/EluElurnn/while/sru_cell/add*
T0
^
rnn/while/sru_cell/mul_2Mulrnn/while/sru_cell/split_1:1rnn/while/sru_cell/Elu*
T0
\
rnn/while/sru_cell/sub_1/xConst^rnn/while/Identity*
valueB
 *  �?*
dtype0
b
rnn/while/sru_cell/sub_1Subrnn/while/sru_cell/sub_1/xrnn/while/sru_cell/split_1:1*
T0
^
rnn/while/sru_cell/mul_3Mulrnn/while/sru_cell/sub_1rnn/while/sru_cell/split:3*
T0
\
rnn/while/sru_cell/add_1Addrnn/while/sru_cell/mul_2rnn/while/sru_cell/mul_3*
T0
�
-rnn/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV33rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enterrnn/while/Identity_1rnn/while/sru_cell/add_1rnn/while/Identity_2*
T0*+
_class!
loc:@rnn/while/sru_cell/add_1
�
3rnn/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnterrnn/TensorArray*
parallel_iterations *'

frame_namernn/while/while_context*
T0*
is_constant(*+
_class!
loc:@rnn/while/sru_cell/add_1
P
rnn/while/add_1/yConst^rnn/while/Identity*
value	B :*
dtype0
H
rnn/while/add_1Addrnn/while/Identity_1rnn/while/add_1/y*
T0
@
rnn/while/NextIterationNextIterationrnn/while/add*
T0
D
rnn/while/NextIteration_1NextIterationrnn/while/add_1*
T0
b
rnn/while/NextIteration_2NextIteration-rnn/while/TensorArrayWrite/TensorArrayWriteV3*
T0
K
rnn/while/NextIteration_3NextIterationrnn/while/sru_cell/add*
T0
5
rnn/while/Exit_2Exitrnn/while/Switch_2*
T0
5
rnn/while/Exit_3Exitrnn/while/Switch_3*
T0
�
&rnn/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3rnn/TensorArrayrnn/while/Exit_2*"
_class
loc:@rnn/TensorArray
n
 rnn/TensorArrayStack/range/startConst*
value	B : *"
_class
loc:@rnn/TensorArray*
dtype0
n
 rnn/TensorArrayStack/range/deltaConst*
value	B :*"
_class
loc:@rnn/TensorArray*
dtype0
�
rnn/TensorArrayStack/rangeRange rnn/TensorArrayStack/range/start&rnn/TensorArrayStack/TensorArraySizeV3 rnn/TensorArrayStack/range/delta*

Tidx0*"
_class
loc:@rnn/TensorArray
�
(rnn/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3rnn/TensorArrayrnn/TensorArrayStack/rangernn/while/Exit_2*
element_shape
:*"
_class
loc:@rnn/TensorArray*
dtype0
E
concatIdentity(rnn/TensorArrayStack/TensorArrayGatherV3*
T0
/
concat_1Identityrnn/while/Exit_3*
T0 