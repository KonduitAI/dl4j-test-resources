
�
in_0Const*�
value�B�
"�      �?                  �?          �?                  �?      �?                          �?  �?                      �?                  �?          �?    *
dtype0
=
	in_0/readIdentityin_0*
T0*
_class
	loc:@in_0
�
in_1Const*�
value�B�
"��E(�5�X��eѾ�==�hx�>�Gr��(r>�����?1�2? c����>�� �\�x>*�@�U�>N=���=�?�M�?�4w���A>���dx�>��?堎>�@��>7Bm=�o��1��?Ȫ����$?Ց'?�)�m_�.��==��I>�ǚ>8v?*
dtype0
=
	in_1/readIdentityin_1*
T0*
_class
	loc:@in_1
Y
in_2Const*=
value4B2
"(            h�e>        ���=    \��>    *
dtype0
=
	in_2/readIdentityin_2*
T0*
_class
	loc:@in_2
S
/softmax_cross_entropy_loss/labels_stop_gradientStopGradient	in_0/read*
T0
R
(softmax_cross_entropy_loss/xentropy/RankConst*
value	B :*
dtype0
^
)softmax_cross_entropy_loss/xentropy/ShapeConst*
valueB"
      *
dtype0
T
*softmax_cross_entropy_loss/xentropy/Rank_1Const*
value	B :*
dtype0
`
+softmax_cross_entropy_loss/xentropy/Shape_1Const*
dtype0*
valueB"
      
S
)softmax_cross_entropy_loss/xentropy/Sub/yConst*
value	B :*
dtype0
�
'softmax_cross_entropy_loss/xentropy/SubSub*softmax_cross_entropy_loss/xentropy/Rank_1)softmax_cross_entropy_loss/xentropy/Sub/y*
T0
~
/softmax_cross_entropy_loss/xentropy/Slice/beginPack'softmax_cross_entropy_loss/xentropy/Sub*
T0*

axis *
N
\
.softmax_cross_entropy_loss/xentropy/Slice/sizeConst*
valueB:*
dtype0
�
)softmax_cross_entropy_loss/xentropy/SliceSlice+softmax_cross_entropy_loss/xentropy/Shape_1/softmax_cross_entropy_loss/xentropy/Slice/begin.softmax_cross_entropy_loss/xentropy/Slice/size*
T0*
Index0
j
3softmax_cross_entropy_loss/xentropy/concat/values_0Const*
valueB:
���������*
dtype0
Y
/softmax_cross_entropy_loss/xentropy/concat/axisConst*
value	B : *
dtype0
�
*softmax_cross_entropy_loss/xentropy/concatConcatV23softmax_cross_entropy_loss/xentropy/concat/values_0)softmax_cross_entropy_loss/xentropy/Slice/softmax_cross_entropy_loss/xentropy/concat/axis*
T0*
N*

Tidx0
�
+softmax_cross_entropy_loss/xentropy/ReshapeReshape	in_1/read*softmax_cross_entropy_loss/xentropy/concat*
T0*
Tshape0
T
*softmax_cross_entropy_loss/xentropy/Rank_2Const*
value	B :*
dtype0
`
+softmax_cross_entropy_loss/xentropy/Shape_2Const*
valueB"
      *
dtype0
U
+softmax_cross_entropy_loss/xentropy/Sub_1/yConst*
value	B :*
dtype0
�
)softmax_cross_entropy_loss/xentropy/Sub_1Sub*softmax_cross_entropy_loss/xentropy/Rank_2+softmax_cross_entropy_loss/xentropy/Sub_1/y*
T0
�
1softmax_cross_entropy_loss/xentropy/Slice_1/beginPack)softmax_cross_entropy_loss/xentropy/Sub_1*
N*
T0*

axis 
^
0softmax_cross_entropy_loss/xentropy/Slice_1/sizeConst*
valueB:*
dtype0
�
+softmax_cross_entropy_loss/xentropy/Slice_1Slice+softmax_cross_entropy_loss/xentropy/Shape_21softmax_cross_entropy_loss/xentropy/Slice_1/begin0softmax_cross_entropy_loss/xentropy/Slice_1/size*
T0*
Index0
l
5softmax_cross_entropy_loss/xentropy/concat_1/values_0Const*
valueB:
���������*
dtype0
[
1softmax_cross_entropy_loss/xentropy/concat_1/axisConst*
dtype0*
value	B : 
�
,softmax_cross_entropy_loss/xentropy/concat_1ConcatV25softmax_cross_entropy_loss/xentropy/concat_1/values_0+softmax_cross_entropy_loss/xentropy/Slice_11softmax_cross_entropy_loss/xentropy/concat_1/axis*
T0*
N*

Tidx0
�
-softmax_cross_entropy_loss/xentropy/Reshape_1Reshape/softmax_cross_entropy_loss/labels_stop_gradient,softmax_cross_entropy_loss/xentropy/concat_1*
T0*
Tshape0
�
#softmax_cross_entropy_loss/xentropySoftmaxCrossEntropyWithLogits+softmax_cross_entropy_loss/xentropy/Reshape-softmax_cross_entropy_loss/xentropy/Reshape_1*
T0
U
+softmax_cross_entropy_loss/xentropy/Sub_2/yConst*
value	B :*
dtype0
�
)softmax_cross_entropy_loss/xentropy/Sub_2Sub(softmax_cross_entropy_loss/xentropy/Rank+softmax_cross_entropy_loss/xentropy/Sub_2/y*
T0
_
1softmax_cross_entropy_loss/xentropy/Slice_2/beginConst*
valueB: *
dtype0
�
0softmax_cross_entropy_loss/xentropy/Slice_2/sizePack)softmax_cross_entropy_loss/xentropy/Sub_2*
N*
T0*

axis 
�
+softmax_cross_entropy_loss/xentropy/Slice_2Slice)softmax_cross_entropy_loss/xentropy/Shape1softmax_cross_entropy_loss/xentropy/Slice_2/begin0softmax_cross_entropy_loss/xentropy/Slice_2/size*
T0*
Index0
�
-softmax_cross_entropy_loss/xentropy/Reshape_2Reshape#softmax_cross_entropy_loss/xentropy+softmax_cross_entropy_loss/xentropy/Slice_2*
T0*
Tshape0
Q
Isoftmax_cross_entropy_loss/assert_broadcastable/static_dims_check_successNoOp
�
softmax_cross_entropy_loss/MulMul-softmax_cross_entropy_loss/xentropy/Reshape_2	in_2/readJ^softmax_cross_entropy_loss/assert_broadcastable/static_dims_check_success*
T0
�
 softmax_cross_entropy_loss/ConstConstJ^softmax_cross_entropy_loss/assert_broadcastable/static_dims_check_success*
valueB: *
dtype0
�
softmax_cross_entropy_loss/SumSumsoftmax_cross_entropy_loss/Mul softmax_cross_entropy_loss/Const*
T0*
	keep_dims( *

Tidx0
�
'softmax_cross_entropy_loss/num_elementsConstJ^softmax_cross_entropy_loss/assert_broadcastable/static_dims_check_success*
value	B :
*
dtype0
�
,softmax_cross_entropy_loss/num_elements/CastCast'softmax_cross_entropy_loss/num_elements*

SrcT0*
Truncate( *

DstT0
�
"softmax_cross_entropy_loss/Const_1ConstJ^softmax_cross_entropy_loss/assert_broadcastable/static_dims_check_success*
valueB *
dtype0
�
 softmax_cross_entropy_loss/Sum_1Sumsoftmax_cross_entropy_loss/Sum"softmax_cross_entropy_loss/Const_1*
	keep_dims( *

Tidx0*
T0
�
 softmax_cross_entropy_loss/valueDivNoNan softmax_cross_entropy_loss/Sum_1,softmax_cross_entropy_loss/num_elements/Cast*
T0 