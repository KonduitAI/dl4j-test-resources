
�
in_0Const*
dtype0*�
value�B�
"�~^G?LM?�p9?�ol>�%:?X�8><q?b|d?��?�al?P@�=,5K?ֹ(?�6?`u#>0�>�{>�h�>�o~?v|?�+-?HM8>v�,?p�e>@�P=��>$T?�q>H�y?hV?(W�>t�>�3?��D? �	?�19?D��> 8�;��=0��=\�W?��??���=��?���>�L?��?��:>�Z$?�j?
=
	in_0/readIdentityin_0*
T0*
_class
	loc:@in_0
A
in_1Const*
dtype0*%
valueB"	   	         
=
	in_1/readIdentityin_1*
T0*
_class
	loc:@in_1
X
embedding_lookup/axisConst*
_class
	loc:@in_0*
dtype0*
value	B : 
�
embedding_lookupGatherV2	in_0/read	in_1/readembedding_lookup/axis*
Taxis0*
Tindices0*
Tparams0*
_class
	loc:@in_0
n
!embedding_lookup/clip_by_norm/mulMulembedding_lookupembedding_lookup*
T0*
_class
	loc:@in_0
z
3embedding_lookup/clip_by_norm/Sum/reduction_indicesConst*
_class
	loc:@in_0*
dtype0*
valueB:
�
!embedding_lookup/clip_by_norm/SumSum!embedding_lookup/clip_by_norm/mul3embedding_lookup/clip_by_norm/Sum/reduction_indices*
T0*

Tidx0*
_class
	loc:@in_0*
	keep_dims(
o
"embedding_lookup/clip_by_norm/SqrtSqrt!embedding_lookup/clip_by_norm/Sum*
T0*
_class
	loc:@in_0
k
%embedding_lookup/clip_by_norm/mul_1/yConst*
_class
	loc:@in_0*
dtype0*
valueB
 *  �?
�
#embedding_lookup/clip_by_norm/mul_1Mulembedding_lookup%embedding_lookup/clip_by_norm/mul_1/y*
T0*
_class
	loc:@in_0
m
'embedding_lookup/clip_by_norm/Maximum/yConst*
_class
	loc:@in_0*
dtype0*
valueB
 *  �?
�
%embedding_lookup/clip_by_norm/MaximumMaximum"embedding_lookup/clip_by_norm/Sqrt'embedding_lookup/clip_by_norm/Maximum/y*
T0*
_class
	loc:@in_0
�
%embedding_lookup/clip_by_norm/truedivRealDiv#embedding_lookup/clip_by_norm/mul_1%embedding_lookup/clip_by_norm/Maximum*
T0*
_class
	loc:@in_0
r
embedding_lookup/clip_by_normIdentity%embedding_lookup/clip_by_norm/truediv*
T0*
_class
	loc:@in_0 