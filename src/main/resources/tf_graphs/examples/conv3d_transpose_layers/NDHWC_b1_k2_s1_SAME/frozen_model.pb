
�
in_0Const*
dtype0*�
value�B�"�~^G?LM?�p9?�ol>�%:?X�8><q?b|d?��?�al?P@�=,5K?ֹ(?�6?`u#>0�>�{>�h�>�o~?v|?�+-?HM8>v�,?p�e>@�P=��>$T?�q>H�y?hV?(W�>t�>�3?��D? �	?�19?D��> 8�;��=0��=\�W?��??���=��?���>�L?��?��:>�Z$?�j?蕚>���>�.>"�V?8�?�W�=��f?�N(?�!�>`Zz>8|�>6�?
�1?���>H]>ut?��>(>ZY	?�\<�'�>�$> ]�>h_~>{�>D?�>l�)?���>��]?H!B>F�@?ޱ8?�A)?�:?z?�{�>4�>? �5=�Ϛ>�U�>rUL?���=��H?��?P�=^CV?�9?���>~�?� \?@t�>��+?Ԃ�>��J>��>�:H?|s�>V?p��=$�T?�:>ZX?��?x��>��?F�*?l�>h�o>r.? Ç>z
?��>�W?���=��>�=,��>(�&> wO>�6�>�.O=�>�>�o?P��=�}�>��p=zB	?��>���>@6>p��=F�e?xĺ>��>D��>6�3?`�>�b?X�2>���>ΣF?��r?�T> k_>^�&?DN�>�Vt?P�@>�5u?��6>�v?N�>�)(?�>V�?RB??���<௖>�X?,?H:?0@I>��j?��x?\�k?@{T=m�> �};��C?��?�A�>4�>H��>�>��N?`J�>?�>p)?�a? �_;���>P"!?�1?�4=V�?�1=h�?��*=�T4?J�|?�Hv?�>Ҋ?�]p?���>�ea=�%?�H�>ZL#?���=`R?�q?��>��?x.�>��?�C�>,I�>�^??�?3?Б�=��#>�Ɖ>�6r?D4?�:?T	??L�R? ��; vK>V�#?^�>D�6?�?�0?8�o? 	E?4LX?�`�>�٠>�g�>�@�>�¼=�ҕ=�E�=�>|?h�m?
=
	in_0/readIdentityin_0*
T0*
_class
	loc:@in_0
�
conv3d_transpose/kernelConst*
dtype0*�
value�B�"���p>�Ͻ>���> f��1�=$��=�!>R�g>g>����3�>m��>>q�>8j ��G?>(%��y�>�5�>pv�<	�>�*پ�����8��t���F\��p�����������Y�>n)��0��=
v
conv3d_transpose/kernel/readIdentityconv3d_transpose/kernel*
T0**
_class 
loc:@conv3d_transpose/kernel
J
conv3d_transpose/biasConst*
valueB"        *
dtype0
p
conv3d_transpose/bias/readIdentityconv3d_transpose/bias*
T0*(
_class
loc:@conv3d_transpose/bias
W
conv3d_transpose/ShapeConst*)
value B"               *
dtype0
R
$conv3d_transpose/strided_slice/stackConst*
valueB: *
dtype0
T
&conv3d_transpose/strided_slice/stack_1Const*
valueB:*
dtype0
T
&conv3d_transpose/strided_slice/stack_2Const*
valueB:*
dtype0
�
conv3d_transpose/strided_sliceStridedSliceconv3d_transpose/Shape$conv3d_transpose/strided_slice/stack&conv3d_transpose/strided_slice/stack_1&conv3d_transpose/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask 
T
&conv3d_transpose/strided_slice_1/stackConst*
valueB:*
dtype0
V
(conv3d_transpose/strided_slice_1/stack_1Const*
valueB:*
dtype0
V
(conv3d_transpose/strided_slice_1/stack_2Const*
dtype0*
valueB:
�
 conv3d_transpose/strided_slice_1StridedSliceconv3d_transpose/Shape&conv3d_transpose/strided_slice_1/stack(conv3d_transpose/strided_slice_1/stack_1(conv3d_transpose/strided_slice_1/stack_2*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
Index0*
T0*
shrink_axis_mask
T
&conv3d_transpose/strided_slice_2/stackConst*
valueB:*
dtype0
V
(conv3d_transpose/strided_slice_2/stack_1Const*
valueB:*
dtype0
V
(conv3d_transpose/strided_slice_2/stack_2Const*
dtype0*
valueB:
�
 conv3d_transpose/strided_slice_2StridedSliceconv3d_transpose/Shape&conv3d_transpose/strided_slice_2/stack(conv3d_transpose/strided_slice_2/stack_1(conv3d_transpose/strided_slice_2/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask 
T
&conv3d_transpose/strided_slice_3/stackConst*
dtype0*
valueB:
V
(conv3d_transpose/strided_slice_3/stack_1Const*
valueB:*
dtype0
V
(conv3d_transpose/strided_slice_3/stack_2Const*
valueB:*
dtype0
�
 conv3d_transpose/strided_slice_3StridedSliceconv3d_transpose/Shape&conv3d_transpose/strided_slice_3/stack(conv3d_transpose/strided_slice_3/stack_1(conv3d_transpose/strided_slice_3/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
Index0*
T0*
shrink_axis_mask
@
conv3d_transpose/mul/yConst*
dtype0*
value	B :
^
conv3d_transpose/mulMul conv3d_transpose/strided_slice_1conv3d_transpose/mul/y*
T0
B
conv3d_transpose/mul_1/yConst*
value	B :*
dtype0
b
conv3d_transpose/mul_1Mul conv3d_transpose/strided_slice_2conv3d_transpose/mul_1/y*
T0
B
conv3d_transpose/mul_2/yConst*
value	B :*
dtype0
b
conv3d_transpose/mul_2Mul conv3d_transpose/strided_slice_3conv3d_transpose/mul_2/y*
T0
B
conv3d_transpose/stack/4Const*
value	B :*
dtype0
�
conv3d_transpose/stackPackconv3d_transpose/strided_sliceconv3d_transpose/mulconv3d_transpose/mul_1conv3d_transpose/mul_2conv3d_transpose/stack/4*
N*
T0*

axis 
�
!conv3d_transpose/conv3d_transposeConv3DBackpropInputV2conv3d_transpose/stackconv3d_transpose/kernel/read	in_0/read*
	dilations	
*
T0*
strides	
*
data_formatNDHWC*
paddingSAME*
Tshape0
�
conv3d_transpose/BiasAddBiasAdd!conv3d_transpose/conv3d_transposeconv3d_transpose/bias/read*
T0*
data_formatNHWC 