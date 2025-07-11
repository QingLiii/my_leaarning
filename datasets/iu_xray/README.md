# IU X-ray 数据集下载说明

## 数据集信息
- **数据集名称**: IU X-ray Dataset
- **Kaggle链接**: https://www.kaggle.com/datasets/jiangzhiyan/iu-xray
- **描述**: 印第安纳大学胸部X光数据集，包含胸部X光图像和对应的放射学报告

## 下载方法

### 方法1: 手动下载（推荐）
1. 访问 https://www.kaggle.com/datasets/jiangzhiyan/iu-xray
2. 点击 "Download" 按钮
3. 将下载的压缩文件解压到当前目录 (`datasets/iu_xray/`)

### 方法2: 使用Kaggle API
如果您想使用Kaggle API自动下载，需要先配置API凭证：

1. **获取API Token**:
   - 登录 Kaggle 网站
   - 进入 Account 设置页面
   - 在 API 部分点击 "Create New API Token"
   - 下载 `kaggle.json` 文件

2. **配置API凭证**:
   - 在用户目录下创建 `.kaggle` 文件夹: `C:\Users\Lu\.kaggle\`
   - 将 `kaggle.json` 文件放入该文件夹
   - 确保文件权限正确

3. **使用命令下载**:
   ```bash
   C:\Users\Lu\AppData\Roaming\Python\Python312\Scripts\kaggle.exe datasets download -d jiangzhiyan/iu-xray -p datasets\iu_xray
   ```

## 数据集结构
下载完成后，数据集应包含以下内容：
- 胸部X光图像文件
- 对应的医学报告文本
- 数据集说明文档

## 注意事项
- 该数据集用于学术研究目的
- 请遵守数据集的使用条款
- 数据集较大，请确保有足够的存储空间