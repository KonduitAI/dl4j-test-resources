
�
in_0Const*m
valuedBb"P~^G?LM?�p9?�ol>�%:?X�8><q?b|d?��?�al?P@�=,5K?ֹ(?�6?`u#>0�>�{>�h�>�o~?v|?*
dtype0
=
	in_0/readIdentityin_0*
T0*
_class
	loc:@in_0
2
rnn/RankConst*
value	B :*
dtype0
9
rnn/range/startConst*
value	B :*
dtype0
9
rnn/range/deltaConst*
value	B :*
dtype0
J
	rnn/rangeRangernn/range/startrnn/Rankrnn/range/delta*

Tidx0
H
rnn/concat/values_0Const*
valueB"       *
dtype0
9
rnn/concat/axisConst*
value	B : *
dtype0
e

rnn/concatConcatV2rnn/concat/values_0	rnn/rangernn/concat/axis*

Tidx0*
T0*
N
G
rnn/transpose	Transpose	in_0/read
rnn/concat*
Tperm0*
T0
O
!rnn/GRUBlockCellV2ZeroState/ConstConst*
dtype0*
valueB:
Q
#rnn/GRUBlockCellV2ZeroState/Const_1Const*
valueB:*
dtype0
Q
'rnn/GRUBlockCellV2ZeroState/concat/axisConst*
value	B : *
dtype0
�
"rnn/GRUBlockCellV2ZeroState/concatConcatV2!rnn/GRUBlockCellV2ZeroState/Const#rnn/GRUBlockCellV2ZeroState/Const_1'rnn/GRUBlockCellV2ZeroState/concat/axis*

Tidx0*
T0*
N
T
'rnn/GRUBlockCellV2ZeroState/zeros/ConstConst*
valueB
 *    *
dtype0
�
!rnn/GRUBlockCellV2ZeroState/zerosFill"rnn/GRUBlockCellV2ZeroState/concat'rnn/GRUBlockCellV2ZeroState/zeros/Const*
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
rnn/strided_sliceStridedSlice	rnn/Shapernn/strided_slice/stackrnn/strided_slice/stack_1rnn/strided_slice/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
Index0*
T0*
shrink_axis_mask
2
rnn/timeConst*
value	B : *
dtype0
�
rnn/TensorArrayTensorArrayV3rnn/strided_slice*
element_shape
:*
clear_after_read(*
dynamic_size( *
identical_element_shapes(*/
tensor_array_namernn/dynamic_rnn/output_0*
dtype0
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
*rnn/TensorArrayUnstack/strided_slice/stackConst*
valueB: *
dtype0
Z
,rnn/TensorArrayUnstack/strided_slice/stack_1Const*
valueB:*
dtype0
Z
,rnn/TensorArrayUnstack/strided_slice/stack_2Const*
valueB:*
dtype0
�
$rnn/TensorArrayUnstack/strided_sliceStridedSlicernn/TensorArrayUnstack/Shape*rnn/TensorArrayUnstack/strided_slice/stack,rnn/TensorArrayUnstack/strided_slice/stack_1,rnn/TensorArrayUnstack/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
Index0*
T0
L
"rnn/TensorArrayUnstack/range/startConst*
value	B : *
dtype0
L
"rnn/TensorArrayUnstack/range/deltaConst*
value	B :*
dtype0
�
rnn/TensorArrayUnstack/rangeRange"rnn/TensorArrayUnstack/range/start$rnn/TensorArrayUnstack/strided_slice"rnn/TensorArrayUnstack/range/delta*

Tidx0
�
>rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3rnn/TensorArray_1rnn/TensorArrayUnstack/rangernn/transposernn/TensorArray_1:1*
T0* 
_class
loc:@rnn/transpose
7
rnn/Maximum/xConst*
value	B :*
dtype0
A
rnn/MaximumMaximumrnn/Maximum/xrnn/strided_slice*
T0
?
rnn/MinimumMinimumrnn/strided_slicernn/Maximum*
T0
E
rnn/while/iteration_counterConst*
value	B : *
dtype0
�
rnn/while/EnterEnterrnn/while/iteration_counter*
T0*
is_constant( *
parallel_iterations *'

frame_namernn/while/while_context
�
rnn/while/Enter_1Enterrnn/time*'

frame_namernn/while/while_context*
T0*
is_constant( *
parallel_iterations 
�
rnn/while/Enter_2Enterrnn/TensorArray:1*
T0*
is_constant( *
parallel_iterations *'

frame_namernn/while/while_context
�
rnn/while/Enter_3Enter!rnn/GRUBlockCellV2ZeroState/zeros*
T0*
is_constant( *
parallel_iterations *'

frame_namernn/while/while_context
T
rnn/while/MergeMergernn/while/Enterrnn/while/NextIteration*
T0*
N
Z
rnn/while/Merge_1Mergernn/while/Enter_1rnn/while/NextIteration_1*
T0*
N
Z
rnn/while/Merge_2Mergernn/while/Enter_2rnn/while/NextIteration_2*
T0*
N
Z
rnn/while/Merge_3Mergernn/while/Enter_3rnn/while/NextIteration_3*
T0*
N
F
rnn/while/LessLessrnn/while/Mergernn/while/Less/Enter*
T0
�
rnn/while/Less/EnterEnterrnn/strided_slice*
parallel_iterations *'

frame_namernn/while/while_context*
T0*
is_constant(
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
!rnn/while/TensorArrayReadV3/EnterEnterrnn/TensorArray_1*'

frame_namernn/while/while_context*
T0*
is_constant(*
parallel_iterations 
�
#rnn/while/TensorArrayReadV3/Enter_1Enter>rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant(*
parallel_iterations *'

frame_namernn/while/while_context
�
rnn/gru_cell/gates/kernelConst*�
value�B�"�<�6���R>>(��T�;���S ��(>PJA=`-�<D<u�Dᦾ;d�X�`>@)�>���>�о��H>P��>~\�>� `>0S�n���Jǽ()�P�	�4�2> |.>���>r�?�￾fh?�2=�Ի��>�� �p=I>�$��FW޾SwȾx�=Pq����)���>�!������?dA~>*
dtype0
N
rnn/gru_cell/gates/kernel/readIdentityrnn/gru_cell/gates/kernel*
T0
\
rnn/gru_cell/gates/biasConst*-
value$B""  �?  �?  �?  �?  �?  �?*
dtype0
J
rnn/gru_cell/gates/bias/readIdentityrnn/gru_cell/gates/bias*
T0
�
rnn/gru_cell/candidate/kernelConst*y
valuepBn"`(�R��~
�p�=��/�$Ҿ�^����>�a
>�B�>���>g�>�!���d�>�!�>C4�dr����V��˚?��\9��:��r���`}7=*
dtype0
V
"rnn/gru_cell/candidate/kernel/readIdentityrnn/gru_cell/candidate/kernel*
T0
T
rnn/gru_cell/candidate/biasConst*!
valueB"            *
dtype0
R
 rnn/gru_cell/candidate/bias/readIdentityrnn/gru_cell/candidate/bias*
T0
�
rnn/while/gru_cell/GRUBlockCellGRUBlockCellrnn/while/TensorArrayReadV3rnn/while/Identity_3%rnn/while/gru_cell/GRUBlockCell/Enter'rnn/while/gru_cell/GRUBlockCell/Enter_1'rnn/while/gru_cell/GRUBlockCell/Enter_2'rnn/while/gru_cell/GRUBlockCell/Enter_3*
T0
�
%rnn/while/gru_cell/GRUBlockCell/EnterEnterrnn/gru_cell/gates/kernel/read*'

frame_namernn/while/while_context*
T0*
is_constant(*
parallel_iterations 
�
'rnn/while/gru_cell/GRUBlockCell/Enter_1Enter"rnn/gru_cell/candidate/kernel/read*
parallel_iterations *'

frame_namernn/while/while_context*
T0*
is_constant(
�
'rnn/while/gru_cell/GRUBlockCell/Enter_2Enterrnn/gru_cell/gates/bias/read*
T0*
is_constant(*
parallel_iterations *'

frame_namernn/while/while_context
�
'rnn/while/gru_cell/GRUBlockCell/Enter_3Enter rnn/gru_cell/candidate/bias/read*
parallel_iterations *'

frame_namernn/while/while_context*
T0*
is_constant(
�
-rnn/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV33rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enterrnn/while/Identity_1!rnn/while/gru_cell/GRUBlockCell:3rnn/while/Identity_2*
T0*2
_class(
&$loc:@rnn/while/gru_cell/GRUBlockCell
�
3rnn/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnterrnn/TensorArray*
is_constant(*2
_class(
&$loc:@rnn/while/gru_cell/GRUBlockCell*
parallel_iterations *'

frame_namernn/while/while_context*
T0
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
V
rnn/while/NextIteration_3NextIteration!rnn/while/gru_cell/GRUBlockCell:3*
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
 rnn/TensorArrayStack/range/deltaConst*
dtype0*
value	B :*"
_class
loc:@rnn/TensorArray
�
rnn/TensorArrayStack/rangeRange rnn/TensorArrayStack/range/start&rnn/TensorArrayStack/TensorArraySizeV3 rnn/TensorArrayStack/range/delta*

Tidx0*"
_class
loc:@rnn/TensorArray
�
(rnn/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3rnn/TensorArrayrnn/TensorArrayStack/rangernn/while/Exit_2*"
_class
loc:@rnn/TensorArray*
dtype0*
element_shape
:
4

rnn/Rank_1Const*
value	B :*
dtype0
;
rnn/range_1/startConst*
value	B :*
dtype0
;
rnn/range_1/deltaConst*
dtype0*
value	B :
R
rnn/range_1Rangernn/range_1/start
rnn/Rank_1rnn/range_1/delta*

Tidx0
J
rnn/concat_2/values_0Const*
valueB"       *
dtype0
;
rnn/concat_2/axisConst*
value	B : *
dtype0
m
rnn/concat_2ConcatV2rnn/concat_2/values_0rnn/range_1rnn/concat_2/axis*

Tidx0*
T0*
N
j
rnn/transpose_1	Transpose(rnn/TensorArrayStack/TensorArrayGatherV3rnn/concat_2*
T0*
Tperm0
,
concatIdentityrnn/transpose_1*
T0
/
concat_1Identityrnn/while/Exit_3*
T0 