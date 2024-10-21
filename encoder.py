class Encoder(nn.Module):
	def __init__(self, wave='haar',channel_in=3,channel_out=32,gc=3,bias=True):
		super(Encoder, self).__init__()
		self.dwt = DWTForward(wave=wave)
		self.spatial_attention = SpatialAttention()
		self.channel_attention = ChannelAttention()
		self.temporal_attention = TemporalAttention()
		self.band_attention = BandAttention()
		self.content_analyzer = ContentAnalyzer()
		#self.residual_factor = nn.Parameter(torch.tensor(1.0))  # 可学习的控制因子
		self.conv1 = nn.Conv3d(6, 32, 3, 1, 1, bias=bias)
		self.conv2 = nn.Conv3d(38, 32, 3, 1, 1, bias=bias)
		self.conv3 = nn.Conv3d(32, 3, 3, 1, 1, bias=bias)
		self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

	def forward(self, host, secret):
		# host: (B, C, T, H, W)
		# secret: (B, C, T, H, W)
		# Apply DWT to both host and secret
		host_dwt = self.dwt(host)
		secret_dwt = self.dwt(secret)

		# Apply attention mechanisms to secret before embedding
		spatial_att = self.spatial_attention(host_dwt)
		temporal_att = self.temporal_attention(host_dwt)
		channel_att = self.channel_attention(host_dwt)
		band_att = self.band_attention(host_dwt)
		#使用concat叠加C=3到C=6
		secret_dwt_attention = apply_attention(secret_dwt, spatial_att, temporal_att, channel_att, band_att)

		stego_dwt_feature = torch.cat((host_dwt, secret_dwt_attention), dim=2)
		stego_dwt_feature = self.lrelu(self.conv1(stego_dwt_feature))
		stego_dwt_feature = torch.cat((torch.cat((host_dwt, secret_dwt_attention), dim=2), stego_dwt_feature), dim=2)
		stego_dwt_feature = self.lrelu(self.conv2(stego_dwt_feature))
		stego_dwt_feature = self.lrelu(self.conv3(stego_dwt_feature))

		# Predict residual factor for each frame using content analyzer
		residual_factors = []
		for i in range(host.shape[2]):  # Iterate over time dimension
			# Extract the current frame from the host
			frame = host[:, :, i, :, :]
			# Analyze the content of the frame
			factor = self.content_analyzer(frame)#改进，factor应该和每一帧，以及需要嵌入的内容有关
			residual_factors.append(factor)
		# Stack the factors along the time dimension
		residual_factors = torch.stack(residual_factors, dim=1).squeeze(-1)
		# Residual connection with predicted control factor
		stego_dwt_res = stego_dwt_feature * residual_factors.unsqueeze(1).unsqueeze(-1).unsqueeze(-1) + host

		stego_res = self.dwt(stego_dwt_res)

		return stego_res

class DWTForward(nn.Module):
    def __init__(self, wave='haar'):
        super(DWTForward, self).__init__()
        self.wave = wave
        self.dec_lo, self.dec_hi = pywt.Wavelet(self.wave).dec_lo, pywt.Wavelet(self.wave).dec_hi
        self.dec_lo = torch.Tensor(self.dec_lo).unsqueeze(0).unsqueeze(0)
        self.dec_hi = torch.Tensor(self.dec_hi).unsqueeze(0).unsqueeze(0)
    def forward(self, x):
        # Assuming input shape is (B, C, T, H, W)
        B, C, T, H, W = x.size()
        # 初始化输出张量
        LL = torch.zeros(B, C, T, H // 2, W // 2, device=x.device)
        LH = torch.zeros(B, C, T, H // 2, W // 2, device=x.device)
        HL = torch.zeros(B, C, T, H // 2, W // 2, device=x.device)
        HH = torch.zeros(B, C, T, H // 2, W // 2, device=x.device)
        for i in range(T):
            # 水平方向的小波变换
            lo = F.conv2d(x[:, :, i], self.dec_lo, stride=2, padding=(1, 0))
            hi = F.conv2d(x[:, :, i], self.dec_hi, stride=2, padding=(1, 0))
            # 垂直方向的小波变换
            lo_ = F.conv2d(lo, self.dec_lo.transpose(-1, -2), stride=2, padding=(0, 1))
            hi_ = F.conv2d(hi, self.dec_lo.transpose(-1, -2), stride=2, padding=(0, 1))
            # 更新子带
            LL[:, :, i] = lo_
            LH[:, :, i] = hi_
            HL[:, :, i] = F.conv2d(lo, self.dec_hi.transpose(-1, -2), stride=2, padding=(0, 1))
            HH[:, :, i] = F.conv2d(hi, self.dec_hi.transpose(-1, -2), stride=2, padding=(0, 1))
        # 将四个子带堆叠成一个张量,#shape (B, 4 , C, T, H/2, W/2)
        return torch.stack((LL, LH, HL, HH), dim=1)

class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'Only kernel sizes 3 and 7 are supported'
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class TemporalAttention(nn.Module):
    def __init__(self):
        super(TemporalAttention, self).__init__()
        self.fc = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=(2, 3, 4, 5), keepdim=True)
        max_out, _ = torch.max(x, dim=(2, 3, 4, 5), keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = x.permute(0, 2, 3, 4, 5, 1).contiguous()
        x = self.fc(x)
        x = x.permute(0, 5, 1, 2, 3, 4).contiguous()
        return self.sigmoid(x)

class ChannelAttention(nn.Module):
    def __init__(self):
        super(ChannelAttention, self).__init__()
        self.fc = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=(3, 4, 5), keepdim=True)
        max_out, _ = torch.max(x, dim=(3, 4, 5), keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = x.permute(0, 2, 3, 4, 5, 1).contiguous()
        x = self.fc(x)
        x = x.permute(0, 5, 1, 2, 3, 4).contiguous()
        return self.sigmoid(x)

class BandAttention(nn.Module):
    def __init__(self):
        super(BandAttention, self).__init__()
        self.fc = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=(2, 3, 4, 5), keepdim=True)
        max_out, _ = torch.max(x, dim=(2, 3, 4, 5), keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = x.permute(0, 1, 3, 4, 5, 2).contiguous()
        x = self.fc(x)
        x = x.permute(0, 5, 1, 2, 3, 4).contiguous()
        return self.sigmoid(x)

def apply_attention(x, spatial_att, temporal_att, channel_att, band_att):
    # Apply the attentions,先把注意力规范到均值为1，min-max均值
    x = x * spatial_att * temporal_att * channel_att * band_att
    return x

class ContentAnalyzer(nn.Module):
	def __init__(self):
		super(ContentAnalyzer, self).__init__()
		self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
		self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
		self.fc1 = nn.Linear(16 * 64 * 64, 128)
		self.fc2 = nn.Linear(128, 1)

	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = x.view(x.size(0), -1)
		x = F.relu(self.fc1(x))
		x = torch.sigmoid(self.fc2(x))
		return x

def jpeg_compression(frame, quality=90):
	# JPEG压缩
	frame = frame.permute(0, 2, 3, 1).numpy()  # 调整维度顺序
	frame = (frame * 255).astype(np.uint8)  # 转换为uint8格式
	result = []
	for i in range(frame.shape[0]):
		img = Image.fromarray(frame[i])
		output = io.BytesIO()
		img.save(output, format="JPEG", quality=quality)
		compressed_img = Image.open(output)
		result.append(np.array(compressed_img))
	result = np.stack(result, axis=0)
	result = result.astype(np.float32) / 255.0
	result = torch.from_numpy(result).permute(0, 3, 1, 2)  # 调整维度顺序
	return result


class Noiser(nn.Module):

	def __init__(self, noise_stddev=0.1, dct_quantization=0.5, conv_kernel_size=3, jpeg_quality=90,
				 motion_kernel_size=5):

		super(Noiser, self).__init__()
		self.noise_stddev = noise_stddev
		self.dct_quantization = dct_quantization
		self.conv_kernel_size = conv_kernel_size
		self.jpeg_quality = jpeg_quality
		self.motion_kernel_size = motion_kernel_size
		self.conv = nn.Conv3d(3, 3, kernel_size=self.conv_kernel_size, padding=self.conv_kernel_size // 2, bias=False)
		self.deconv = nn.ConvTranspose3d(3, 3, kernel_size=self.conv_kernel_size, padding=self.conv_kernel_size // 2,
										 bias=False)

	def add_noise(self, x):
		# 添加随机噪声
		noise = torch.randn_like(x) * self.noise_stddev
		return x + noise

	def apply_dct(self, x):
		# 进行 DCT 变换
		x_dct = torch.rfft(x, 2, normalized=False, onesided=True)
		# 去除高频系数
		x_dct[:, :, :, :x_dct.shape[3] // 2, :] *= self.dct_quantization
		x_dct[:, :, :, :, :x_dct.shape[4] // 2] *= self.dct_quantization
		# 进行反变换
		x_idct = torch.irfft(x_dct, 2, normalized=False, onesided=True, signal_sizes=(x.shape[-2], x.shape[-1]))
		return x_idct

	def apply_convolution(self, x):
		# 应用卷积
		x_conv = self.conv(x)
		# 应用反卷积
		x_deconv = self.deconv(x_conv)
		return x_deconv

	def apply_jpeg_compression(self, x):
		# 应用JPEG压缩
		return jpeg_compression(x, self.jpeg_quality)

	def apply_motion_blur(self, x):
		# 应用运动模糊
		return motion_blur(x, self.motion_kernel_size)

	def forward(self, x):
		# 随机选择一种失真类型
		distortion_type = np.random.choice(['noise', 'dct', 'convolution', 'jpeg', 'motion'])
		if distortion_type == 'noise':
			return self.add_noise(x)
		elif distortion_type == 'dct':
			return self.apply_dct(x)
		elif distortion_type == 'convolution':
			return self.apply_convolution(x)
		elif distortion_type == 'jpeg':
			return self.apply_jpeg_compression(x)
		elif distortion_type == 'motion':
			return self.apply_motion_blur(x)