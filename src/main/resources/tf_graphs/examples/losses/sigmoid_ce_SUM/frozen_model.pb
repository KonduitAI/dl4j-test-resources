
e
in_0Const*
dtype0*I
value@B>"0~^G?LM?�p9?�ol>�%:?X�8><q?b|d?��?�al?P@�=,5K?
=
	in_0/readIdentityin_0*
T0*
_class
	loc:@in_0
e
in_1Const*
dtype0*I
value@B>"0����%?6��<��3�->�F�>���l�O4�=���>t��՘~���R>
=
	in_1/readIdentityin_1*
T0*
_class
	loc:@in_1
A
in_2Const*
dtype0*%
valueB"    ���><!r?
=
	in_2/readIdentityin_2*
T0*
_class
	loc:@in_2
c
.sigmoid_cross_entropy_loss/xentropy/zeros_likeConst*
dtype0*
valueB*    
�
0sigmoid_cross_entropy_loss/xentropy/GreaterEqualGreaterEqual	in_1/read.sigmoid_cross_entropy_loss/xentropy/zeros_like*
T0
�
*sigmoid_cross_entropy_loss/xentropy/SelectSelect0sigmoid_cross_entropy_loss/xentropy/GreaterEqual	in_1/read.sigmoid_cross_entropy_loss/xentropy/zeros_like*
T0
B
'sigmoid_cross_entropy_loss/xentropy/NegNeg	in_1/read*
T0
�
,sigmoid_cross_entropy_loss/xentropy/Select_1Select0sigmoid_cross_entropy_loss/xentropy/GreaterEqual'sigmoid_cross_entropy_loss/xentropy/Neg	in_1/read*
T0
M
'sigmoid_cross_entropy_loss/xentropy/mulMul	in_1/read	in_0/read*
T0
�
'sigmoid_cross_entropy_loss/xentropy/subSub*sigmoid_cross_entropy_loss/xentropy/Select'sigmoid_cross_entropy_loss/xentropy/mul*
T0
e
'sigmoid_cross_entropy_loss/xentropy/ExpExp,sigmoid_cross_entropy_loss/xentropy/Select_1*
T0
d
)sigmoid_cross_entropy_loss/xentropy/Log1pLog1p'sigmoid_cross_entropy_loss/xentropy/Exp*
T0
�
#sigmoid_cross_entropy_loss/xentropyAdd'sigmoid_cross_entropy_loss/xentropy/sub)sigmoid_cross_entropy_loss/xentropy/Log1p*
T0
Q
Isigmoid_cross_entropy_loss/assert_broadcastable/static_dims_check_successNoOp
�
sigmoid_cross_entropy_loss/MulMul#sigmoid_cross_entropy_loss/xentropy	in_2/readJ^sigmoid_cross_entropy_loss/assert_broadcastable/static_dims_check_success*
T0
�
 sigmoid_cross_entropy_loss/ConstConstJ^sigmoid_cross_entropy_loss/assert_broadcastable/static_dims_check_success*
dtype0*
valueB"       
�
sigmoid_cross_entropy_loss/SumSumsigmoid_cross_entropy_loss/Mul sigmoid_cross_entropy_loss/Const*
T0*

Tidx0*
	keep_dims(  