
G
imagePlaceholder*
dtype0*$
shape:���������
�
filterW1Const*�
value�B�"��Ɔ?e�d�ܞ>�m�?�!�<�?ο��?��N���=9y���B?�;˘����i��绾Ku�?�K��� ?)�l�p����� �?)=���n�D}.?'�0?(rE���޽c\]?dw��Nƪ>,�ۿ�0>o�����?�w����?�);>QJ�?^�8�D��X�?�k��X��碰>�e&?EeҿT	m?C��=f�>]��fȼ?=�>&�@��U�����{�M?d��?�쁿���d��s]?���>�н>JML��J?������>>		ƿ-��>!_s?��?��]��<���5@Q�?Y�6?kk�?�@�?a;?D˾�$'��7���{��m��>e�h�d�>����~Ͻ��Y�Eb&��@�?ģ�HF�
i��2wݼ�ي?X�'����>�q�X�?�e�0�饿w"�2ݱ?,�f?�꧿?J�݄!��{q�[�>$/�>�g�>Q^9?��>��?�/?=�@��?ld��]?$�^����?��z��?�d&�w%�� �I>��/?��5?y�ƿVt���Х>��O?]�D>i�k�]�"����?['?�20�W%�?cރ?G�t�����V7����$@x࿟��?�tx?%0����?ұK�kݚ=p���	c�?�꠿vG�>�$�I/x?��?[�'?%���=$?AhϾ�N'�C��>n����r=r��`t?���?c�¢����K?�I ?��c�p�?��e��8�?��4>!�N?���?���>�r�x�w?���^�]��dʿ����m<��<���?W�$�=׻>�Y">3-�w�	?�͙�����v?�])��hվ�y�z�@���>mЈ?'��?u�<�q��=�;@�0¾Pd?S����{��]��>/�?���?�*���N?��ؽ����]�?��=�9@*
dtype0
I
filterW1/readIdentityfilterW1*
T0*
_class
loc:@filterW1
�
conv2d1Conv2DimagefilterW1/read*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
x
	avg_pool2AvgPoolconv2d1*
ksize
*
paddingSAME*
T0*
strides
*
data_formatNHWC
�
filterW3Const*�
value�B�"���Y����>#`ܾڇ?ض?)?���<^c?���>w���ԥ�>������= (�>l�?�i����?��Q�"�W?�֚�"߿������?(�[?��$�n?i�����?�_T?Bp�L��b?��G?Rq?A�?9�)@�=�E>�<Щ?Re�?���>9�}:o��ќ?���<L��?�?�?v?����=rL�O��?��׿޸�?Jp�q�T��l	��֝=W�
��C�>f��?Aċ�38�?�/�?�ٷ��T?3D�>���?��/?��?q�N��<8�>�c�>�!9?�y�|V�>�Z¿�G��m^� FN?�69�f�I>]|x�q�	�����nN?*\ ?��n?�@�>����t�>�������4}����>Z�x?����~��>,�Q?\�翱�8����m���DJ�<O?d��hn?�����79�f^�>5پ_U�?"jP���?�>���?,:�?�:�>��>���9{�Md��[��j�� >i�g?����B�"�[�DZ?&M��Y��>L��?t�@'oͿ�?�>�e?$<>I���>�a���%޾F�,���ǽ��?1S @W���J��>��:@7���q,�d�1�6>�H����
�9�
�����2�f��D? �׿*�>�V2?`�G?�7���E?���?a���#n ��{?���q���bT?G�?B�@=��>ř�>ٔC?�]�?�|? $���?�P4>A���Ky���S?�?)>���Aܾf�>A��A=Pw�>�}�>�e@3�O�%ޜ=sy�<g��>Pl�?��>�
>$N�?z�����^�
��>�^�?3e�=�U?�}����v�A���j�?/Xv=
�1?�c�<dh?~4�~俌ω��|Ӿ�;�?�!����c���?R��*
dtype0
I
filterW3/readIdentityfilterW3*
T0*
_class
loc:@filterW3
�
conv2d3Conv2D	avg_pool2filterW3/read*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
	dilations
*
T0
x
	max_pool4MaxPoolconv2d3*
paddingSAME*
T0*
strides
*
data_formatNHWC*
ksize

B
Reshape/shapeConst*
valueB"����   *
dtype0
C
ReshapeReshape	max_pool4Reshape/shape*
T0*
Tshape0
Q
VariableConst*1
value(B&"��=ja?���>?|?�K?�z[?*
dtype0
I
Variable/readIdentityVariable*
T0*
_class
loc:@Variable
W
outputMatMulReshapeVariable/read*
T0*
transpose_a( *
transpose_b(  