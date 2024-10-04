class Worker(QObject):
    # シグナルの定義
    finished = pyqtSignal()
    error = pyqtSignal(str)
    plot_waveform_signal = pyqtSignal(object, object, object, object, bool)
    plot_stft_signal = pyqtSignal(object, object, object, object)
    plot_melspectrogram_signal = pyqtSignal(object, object, object, object)
    plot_zero_crossing_rate_signal = pyqtSignal(object, object, object, object, bool)
    plot_bandpass_filter_signal = pyqtSignal(object, object, object, object)
    plot_pitch_detection_signal = pyqtSignal(object, object, object, object)
    plot_notch_filter_signal = pyqtSignal(object, object, object, object)
    compare_analysis_signal = pyqtSignal(object, object, object, object)
    evaluate_similarity_signal = pyqtSignal(object, object, object, object)
    detect_anomalies_signal = pyqtSignal(object, object, object, object)
    advanced_visualization_signal = pyqtSignal(object, object, object, object)
    show_statistics_signal = pyqtSignal(object, object, object, object)

    def __init__(self, audio_file1, audio_file2, trim_enabled, start_time, end_time, selected_features, overlay):
        super().__init__()
        self.audio_file1 = audio_file1
        self.audio_file2 = audio_file2
        self.trim_enabled = trim_enabled
        self.start_time = start_time
        self.end_time = end_time
        self.selected_features = selected_features
        self.overlay = overlay

    def run(self):
        try:
            y1, sr1 = None, None
            y2, sr2 = None, None

            # 音声ファイル1の読み込みとトリミング
            if self.audio_file1:
                if self.trim_enabled:
                    y1, sr1 = self.load_and_trim(self.audio_file1, self.start_time, self.end_time)
                else:
                    y1, sr1 = librosa.load(self.audio_file1, sr=None)
                # 可視化用にダウンサンプリング
                y1, sr1 = self.downsample(y1, sr1, factor=4)

            # 音声ファイル2の読み込みとトリミング
            if self.audio_file2:
                if self.trim_enabled:
                    y2, sr2 = self.load_and_trim(self.audio_file2, self.start_time, self.end_time)
                else:
                    y2, sr2 = librosa.load(self.audio_file2, sr=None)
                # 可視化用にダウンサンプリング
                y2, sr2 = self.downsample(y2, sr2, factor=4)

            # 選択された特徴量ごとにシグナルを発行
            for feature in self.selected_features:
                if feature == "波形":
                    self.plot_waveform_signal.emit(y1, sr1, y2, sr2, self.overlay)
                elif feature == "STFT":
                    self.plot_stft_signal.emit(y1, sr1, y2, sr2)
                elif feature == "メルスペクトログラム":
                    self.plot_melspectrogram_signal.emit(y1, sr1, y2, sr2)
                elif feature == "ゼロ交差率":
                    self.plot_zero_crossing_rate_signal.emit(y1, sr1, y2, sr2, self.overlay)
                elif feature == "バンドパスフィルタ":
                    self.plot_bandpass_filter_signal.emit(y1, sr1, y2, sr2)
                elif feature == "ピッチ検出":
                    self.plot_pitch_detection_signal.emit(y1, sr1, y2, sr2)
                elif feature == "ノッチフィルタ":
                    self.plot_notch_filter_signal.emit(y1, sr1, y2, sr2)
                elif feature == "比較分析":
                    self.compare_analysis_signal.emit(y1, sr1, y2, sr2)
                elif feature == "類似度評価":
                    self.evaluate_similarity_signal.emit(y1, sr1, y2, sr2)
                elif feature == "異常検知":
                    self.detect_anomalies_signal.emit(y1, sr1, y2, sr2)
                elif feature == "高度な可視化":
                    self.advanced_visualization_signal.emit(y1, sr1, y2, sr2)
                elif feature == "統計情報":
                    self.show_statistics_signal.emit(y1, sr1, y2, sr2)
                else:
                    pass  # 未対応の特徴量の場合はスキップ

            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))

    def load_and_trim(self, file_path, start_time=0.0, end_time=0.0):
        y, sr = librosa.load(file_path, sr=None)
        total_duration = librosa.get_duration(y=y, sr=sr)
        if end_time == 0.0 or end_time > total_duration:
            end_time = total_duration
        if start_time >= end_time:
            raise ValueError("開始時間は終了時間より小さくする必要があります。")
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        y_trimmed = y[start_sample:end_sample]
        return y_trimmed, sr

    def downsample(self, y, sr, factor=4):
        if factor <= 1:
            return y, sr
        y_down = librosa.resample(y, orig_sr=sr, target_sr=sr // factor)
        sr_down = sr // factor
        return y_down, sr_down

class AudioFeatureApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('オーディオ特徴抽出アプリ (MP3/WAV)')
        self.setGeometry(100, 100, 1600, 900)  # ウィンドウサイズを大きく
        self.initUI()

        # 分析データの保存用
        self.analysis_data = {}

    def initUI(self):
        # レイアウト
        main_layout = QHBoxLayout()
        sidebar_layout = QVBoxLayout()
        content_layout = QVBoxLayout()

        # サイドバーのコンポーネント
        sidebar_title = QLabel("メニュー")
        sidebar_title.setStyleSheet("font-size: 18px; font-weight: bold;")
        sidebar_layout.addWidget(sidebar_title)

        # ファイルアップロードボタン
        upload_btn1 = QPushButton("音声ファイル1を選択")
        upload_btn1.clicked.connect(self.upload_file1)
        self.file_label1 = QLabel("選択されていません")

        upload_btn2 = QPushButton("音声ファイル2を選択")
        upload_btn2.clicked.connect(self.upload_file2)
        self.file_label2 = QLabel("選択されていません")

        sidebar_layout.addWidget(upload_btn1)
        sidebar_layout.addWidget(self.file_label1)
        sidebar_layout.addWidget(upload_btn2)
        sidebar_layout.addWidget(self.file_label2)

        # トリミングオプション
        trim_checkbox = QCheckBox("音声を切り取る")
        trim_checkbox.stateChanged.connect(self.toggle_trimming)
        self.trim_group = QGroupBox("音声の切り取り設定")
        trim_layout = QGridLayout()

        self.start_time_input = QLineEdit("0.0")
        self.end_time_input = QLineEdit("0.0")
        trim_layout.addWidget(QLabel("開始時間（秒）"), 0, 0)
        trim_layout.addWidget(self.start_time_input, 0, 1)
        trim_layout.addWidget(QLabel("終了時間（秒）"), 1, 0)
        trim_layout.addWidget(self.end_time_input, 1, 1)
        self.trim_group.setLayout(trim_layout)
        self.trim_group.setEnabled(False)

        sidebar_layout.addWidget(trim_checkbox)
        sidebar_layout.addWidget(self.trim_group)

        # 特徴選択
        feature_label = QLabel("抽出して可視化する特徴を選択してください：")
        self.feature_list = QListWidget()
        self.feature_list.setSelectionMode(QListWidget.MultiSelection)
        features = [
            "波形", "STFT", "メルスペクトログラム", "ゼロ交差率",
            "バンドパスフィルタ", "ピッチ検出", "ノッチフィルタ",
            "比較分析", "異常検知", "類似度評価", "高度な可視化",
            "統計情報"
        ]
        self.feature_list.addItems(features)
        sidebar_layout.addWidget(feature_label)
        sidebar_layout.addWidget(self.feature_list)

        # オーバーレイオプション
        self.overlay_checkbox = QCheckBox("重ね合わせ表示（比較可能な場合）")
        sidebar_layout.addWidget(self.overlay_checkbox)

        # 処理開始ボタン
        process_btn = QPushButton("処理を開始")
        process_btn.clicked.connect(self.start_processing)
        sidebar_layout.addWidget(process_btn)

        # 画像保存ボタン
        save_images_btn = QPushButton("画像を保存")
        save_images_btn.clicked.connect(self.save_images)
        save_images_btn.setEnabled(False)  # 初期状態では無効
        self.save_images_btn = save_images_btn  # 後で有効化するために保持
        sidebar_layout.addWidget(save_images_btn)

        # 処理結果表示エリア
        result_label = QLabel("処理結果:")
        self.result_area = QTextEdit()
        self.result_area.setReadOnly(True)
        self.result_area.setFixedHeight(200)  # 固定高さを設定
        sidebar_layout.addWidget(result_label)
        sidebar_layout.addWidget(self.result_area)

        # サイドバーのレイアウトに伸縮を加える
        sidebar_layout.addStretch()

        # コンテンツエリア（スクロール可能でプロットのみ）
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout()
        self.content_widget.setLayout(self.content_layout)
        scroll.setWidget(self.content_widget)
        content_layout.addWidget(scroll)

        # レイアウトの組み立て
        main_layout.addLayout(sidebar_layout, 1)  # サイドバーを小さく
        main_layout.addLayout(content_layout, 4)  # コンテンツエリアを大きく
        self.setLayout(main_layout)

        # 初期化
        self.audio_file1 = None
        self.audio_file2 = None
        self.figures = []  # 保存用のFigureオブジェクトを保持

    def toggle_trimming(self, state):
        self.trim_group.setEnabled(state == Qt.Checked)

    def upload_file1(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "音声ファイル1を選択", "", "Audio Files (*.wav *.mp3)", options=options)
        if fileName:
            self.file_label1.setText(os.path.basename(fileName))
            self.audio_file1 = fileName

    def upload_file2(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "音声ファイル2を選択", "", "Audio Files (*.wav *.mp3)", options=options)
        if fileName:
            self.file_label2.setText(os.path.basename(fileName))
            self.audio_file2 = fileName

    def start_processing(self):
        selected_features = [item.text() for item in self.feature_list.selectedItems()]
        overlay = self.overlay_checkbox.isChecked()

        # 入力の検証
        if not selected_features:
            QMessageBox.warning(self, "警告", "少なくとも1つの特徴を選択してください。")
            return

        if self.audio_file1 is None and self.audio_file2 is None:
            QMessageBox.warning(self, "警告", "少なくとも1つの音声ファイルをアップロードしてください。")
            return

        # トリミング設定の取得
        trim_enabled = self.trim_group.isEnabled()
        try:
            if trim_enabled:
                start_time = float(self.start_time_input.text())
                end_time = float(self.end_time_input.text())
            else:
                start_time, end_time = 0.0, 0.0
        except ValueError:
            QMessageBox.warning(self, "警告", "開始時間と終了時間は数値で入力してください。")
            return

        # 以前のコンテンツをクリア
        for i in reversed(range(self.content_layout.count())):
            widget_to_remove = self.content_layout.itemAt(i).widget()
            if widget_to_remove is not None:
                self.content_layout.removeWidget(widget_to_remove)
                widget_to_remove.setParent(None)

        # 分析データのリセット
        self.analysis_data = {}
        self.figures = []

        # 処理ボタンと保存ボタンを無効化
        sender = self.sender()
        sender.setEnabled(False)
        self.save_images_btn.setEnabled(False)  # 保存ボタンも無効化

        # 処理中メッセージを表示
        self.result_area.append("処理中...")

        # スレッドの作成
        self.thread = QThread()
        # ワーカーオブジェクトの作成
        self.worker = Worker(
            audio_file1=self.audio_file1,
            audio_file2=self.audio_file2,
            trim_enabled=trim_enabled,
            start_time=start_time,
            end_time=end_time,
            selected_features=selected_features,
            overlay=overlay
        )
        # ワーカーをスレッドに移動
        self.worker.moveToThread(self.thread)
        # シグナルとスロットの接続
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.error.connect(self.show_error)
        # プロットシグナルの接続
        self.worker.plot_waveform_signal.connect(self.handle_plot_waveform)
        self.worker.plot_stft_signal.connect(self.handle_plot_stft)
        self.worker.plot_melspectrogram_signal.connect(self.handle_plot_melspectrogram)
        self.worker.plot_zero_crossing_rate_signal.connect(self.handle_plot_zero_crossing_rate)
        self.worker.plot_bandpass_filter_signal.connect(self.handle_plot_bandpass_filter)
        self.worker.plot_pitch_detection_signal.connect(self.handle_plot_pitch_detection)
        self.worker.plot_notch_filter_signal.connect(self.handle_plot_notch_filter)
        self.worker.compare_analysis_signal.connect(self.handle_compare_analysis)
        self.worker.evaluate_similarity_signal.connect(self.handle_evaluate_similarity)
        self.worker.detect_anomalies_signal.connect(self.handle_detect_anomalies)
        self.worker.advanced_visualization_signal.connect(self.handle_advanced_visualization)
        self.worker.show_statistics_signal.connect(self.handle_show_statistics)
        # 他の特徴量のシグナル接続もここに追加

        # スレッドの開始
        self.thread.start()

        # 処理完了時に処理ボタンを再度有効化
        self.thread.finished.connect(lambda: sender.setEnabled(True))
        # 処理完了時に保存ボタンを有効化
        self.thread.finished.connect(lambda: self.save_images_btn.setEnabled(True))
        # 処理完了メッセージを表示
        self.thread.finished.connect(lambda: self.result_area.append("処理が完了しました。"))

    def show_error(self, message):
        QMessageBox.critical(self, "エラー", message)
        # 処理ボタンと保存ボタンを再度有効化
        sender = self.sender()
        if isinstance(sender, QPushButton):
            sender.setEnabled(True)
        self.save_images_btn.setEnabled(False)
        self.result_area.append(f"エラー: {message}")

    def handle_plot_waveform(self, y1, sr1, y2, sr2, overlay):
        # 波形のプロット
        if y1 is None:
            return

        fig, ax = plt.subplots(figsize=(12, 4))  # プロットを大きく
        librosa.display.waveshow(y1, sr=sr1, alpha=0.5, label='音声ファイル1')

        if y2 is not None:
            if overlay:
                librosa.display.waveshow(y2, sr=sr2, color='r', alpha=0.5, label='音声ファイル2')
                ax.set_title("波形の重ね合わせ表示")
                ax.legend()
            else:
                ax.set_title("音声ファイル1の波形")
        else:
            ax.set_title("音声ファイル1の波形")
        
        # 分析データの保存
        self.analysis_data['波形_音声ファイル1'] = y1.tolist()
        if y2 is not None:
            self.analysis_data['波形_音声ファイル2'] = y2.tolist()

        canvas = FigureCanvas(fig)
        self.content_layout.addWidget(canvas)
        self.figures.append(fig)  # 保存用

        if y2 is not None and not overlay:
            fig2, ax2 = plt.subplots(figsize=(12, 4))
            librosa.display.waveshow(y2, sr=sr2, color='r')
            ax2.set_title("音声ファイル2の波形")
            ax2.legend(['音声ファイル2'])
            self.analysis_data['波形_音声ファイル2'] = y2.tolist()
            canvas2 = FigureCanvas(fig2)
            self.content_layout.addWidget(canvas2)
            self.figures.append(fig2)  # 保存用

    def handle_plot_stft(self, y1, sr1, y2, sr2):
        # STFTのプロット
        if y1 is None:
            return

        fig, ax = plt.subplots(figsize=(12, 6))
        D1 = librosa.stft(y1)
        S_db1 = librosa.amplitude_to_db(np.abs(D1), ref=np.max)
        img1 = librosa.display.specshow(S_db1, sr=sr1, x_axis='time', y_axis='log', ax=ax, cmap='viridis')
        fig.colorbar(img1, ax=ax, format='%+2.0f dB')
        ax.set_title('音声ファイル1のSTFT')

        # 分析データの保存
        self.analysis_data['STFT_音声ファイル1'] = S_db1.tolist()

        if y2 is not None:
            D2 = librosa.stft(y2)
            S_db2 = librosa.amplitude_to_db(np.abs(D2), ref=np.max)
            img2 = librosa.display.specshow(S_db2, sr=sr2, x_axis='time', y_axis='log', ax=ax, alpha=0.5, cmap='Reds')
            ax.set_title('STFT')
            self.analysis_data['STFT_音声ファイル2'] = S_db2.tolist()

        canvas = FigureCanvas(fig)
        self.content_layout.addWidget(canvas)
        self.figures.append(fig)  # 保存用

    def handle_plot_melspectrogram(self, y1, sr1, y2, sr2):
        # メルスペクトログラムのプロット
        if y1 is None:
            return

        fig, ax = plt.subplots(figsize=(12, 6))
        S1 = librosa.feature.melspectrogram(y=y1, sr=sr1, n_mels=128, fmax=8000)
        S_dB1 = librosa.power_to_db(S1, ref=np.max)
        img1 = librosa.display.specshow(S_dB1, sr=sr1, x_axis='time', y_axis='mel', fmax=8000, ax=ax, cmap='plasma')
        fig.colorbar(img1, ax=ax, format='%+2.0f dB')
        ax.set_title('音声ファイル1のメルスペクトログラム')

        # 分析データの保存
        self.analysis_data['メルスペクトログラム_音声ファイル1'] = S_dB1.tolist()

        if y2 is not None:
            S2 = librosa.feature.melspectrogram(y=y2, sr=sr2, n_mels=128, fmax=8000)
            S_dB2 = librosa.power_to_db(S2, ref=np.max)
            img2 = librosa.display.specshow(S_dB2, sr=sr2, x_axis='time', y_axis='mel', fmax=8000, ax=ax, alpha=0.5, cmap='inferno')
            ax.set_title('メルスペクトログラム')
            self.analysis_data['メルスペクトログラム_音声ファイル2'] = S_dB2.tolist()

        canvas = FigureCanvas(fig)
        self.content_layout.addWidget(canvas)
        self.figures.append(fig)  # 保存用

    def handle_plot_zero_crossing_rate(self, y1, sr1, y2, sr2, overlay):
        # ゼロ交差率のプロット
        if y1 is None:
            return

        fig, ax = plt.subplots(figsize=(12, 4))
        zcr1 = librosa.feature.zero_crossing_rate(y1)[0]
        frames1 = range(len(zcr1))
        times1 = librosa.frames_to_time(frames1, sr=sr1)
        ax.plot(times1, zcr1, label='音声ファイル1')

        # 分析データの保存
        self.analysis_data['ゼロ交差率_音声ファイル1'] = zcr1.tolist()

        if y2 is not None:
            zcr2 = librosa.feature.zero_crossing_rate(y2)[0]
            frames2 = range(len(zcr2))
            times2 = librosa.frames_to_time(frames2, sr=sr2)
            if overlay:
                ax.plot(times2, zcr2, label='音声ファイル2', alpha=0.7)
                ax.set_title("ゼロ交差率の重ね合わせ表示")
            else:
                ax.plot(times2, zcr2, color='r', label='音声ファイル2')
                ax.set_title("ゼロ交差率")
            self.analysis_data['ゼロ交差率_音声ファイル2'] = zcr2.tolist()
        else:
            ax.set_title("ゼロ交差率")

        ax.set_xlabel("時間 (秒)")
        ax.set_ylabel("ゼロ交差率")
        ax.legend()
        canvas = FigureCanvas(fig)
        self.content_layout.addWidget(canvas)
        self.figures.append(fig)  # 保存用

    def handle_plot_bandpass_filter(self, y1, sr1, y2, sr2):
        # バンドパスフィルタ適用後の波形のプロット（500Hz - 1000Hz）
        if y1 is None:
            return

        # バンドパスフィルタの設定
        low_freq = 500
        high_freq = 1000
        nyquist1 = 0.5 * sr1
        low = low_freq / nyquist1
        high = high_freq / nyquist1
        b1, a1 = butter(4, [low, high], btype='band')
        filtered_signal1 = lfilter(b1, a1, y1)
        fig, ax = plt.subplots(figsize=(12, 4))
        librosa.display.waveshow(filtered_signal1, sr=sr1, ax=ax, alpha=0.7, label='音声ファイル1')
        ax.set_title(f"音声ファイル1のバンドパスフィルタ適用信号（{low_freq}Hz - {high_freq}Hz）")

        # 分析データの保存
        self.analysis_data['バンドパスフィルタ_音声ファイル1'] = filtered_signal1.tolist()

        if y2 is not None:
            nyquist2 = 0.5 * sr2
            low2 = low_freq / nyquist2
            high2 = high_freq / nyquist2
            b2, a2 = butter(4, [low2, high2], btype='band')
            filtered_signal2 = lfilter(b2, a2, y2)
            librosa.display.waveshow(filtered_signal2, sr=sr2, color='r', ax=ax, alpha=0.7, label='音声ファイル2')
            ax.set_title(f"バンドパスフィルタ適用信号（{low_freq}Hz - {high_freq}Hz）")
            ax.legend(['音声ファイル1', '音声ファイル2'])
            self.analysis_data['バンドパスフィルタ_音声ファイル2'] = filtered_signal2.tolist()
        else:
            ax.legend(['音声ファイル1'])

        canvas = FigureCanvas(fig)
        self.content_layout.addWidget(canvas)
        self.figures.append(fig)  # 保存用

    def handle_plot_pitch_detection(self, y1, sr1, y2, sr2):
        # ピッチ検出のプロット
        if y1 is None:
            return

        fig, ax = plt.subplots(figsize=(12, 4))
        pitches1, magnitudes1 = librosa.piptrack(y=y1, sr=sr1)
        pitch_values1 = [
            pitches1[:, t][magnitudes1[:, t].argmax()] if magnitudes1[:, t].max() > 0 else np.nan
            for t in range(pitches1.shape[1])
        ]
        times1 = librosa.frames_to_time(range(len(pitch_values1)), sr=sr1)
        ax.plot(times1, pitch_values1, label='音声ファイル1')

        # 分析データの保存
        self.analysis_data['ピッチ検出_音声ファイル1'] = pitch_values1

        if y2 is not None:
            pitches2, magnitudes2 = librosa.piptrack(y=y2, sr=sr2)
            pitch_values2 = [
                pitches2[:, t][magnitudes2[:, t].argmax()] if magnitudes2[:, t].max() > 0 else np.nan
                for t in range(pitches2.shape[1])
            ]
            times2 = librosa.frames_to_time(range(len(pitch_values2)), sr=sr2)
            if self.overlay_checkbox.isChecked():
                ax.plot(times2, pitch_values2, label='音声ファイル2', alpha=0.7)
                ax.set_title("ピッチトラッキングの重ね合わせ表示")
            else:
                ax.plot(times2, pitch_values2, color='r', label='音声ファイル2')
                ax.set_title("ピッチトラッキング")
            self.analysis_data['ピッチ検出_音声ファイル2'] = pitch_values2
        else:
            ax.set_title("音声ファイル1のピッチトラッキング")

        ax.set_xlabel("時間 (秒)")
        ax.set_ylabel("周波数 (Hz)")
        ax.legend()
        canvas = FigureCanvas(fig)
        self.content_layout.addWidget(canvas)
        self.figures.append(fig)  # 保存用

    def handle_plot_notch_filter(self, y1, sr1, y2, sr2):
        # ノッチフィルタ適用後の波形のプロット
        if y1 is None:
            return

        # ノッチフィルタの設定
        remove_freq = 60  # 60Hzを除去（例）
        q_factor = 30.0    # Q値（例）

        fig, ax = plt.subplots(figsize=(12, 4))
        nyquist1 = 0.5 * sr1
        notch_freq1 = remove_freq / nyquist1
        b1, a1 = iirnotch(notch_freq1, q_factor)
        filtered_signal1 = lfilter(b1, a1, y1)
        librosa.display.waveshow(filtered_signal1, sr=sr1, ax=ax, alpha=0.7, label='音声ファイル1')
        ax.set_title(f"音声ファイル1のノッチフィルタ適用信号（{remove_freq}Hzを除去）")

        # 分析データの保存
        self.analysis_data['ノッチフィルタ_音声ファイル1'] = filtered_signal1.tolist()

        if y2 is not None:
            nyquist2 = 0.5 * sr2
            notch_freq2 = remove_freq / nyquist2
            b2, a2 = iirnotch(notch_freq2, q_factor)
            filtered_signal2 = lfilter(b2, a2, y2)
            librosa.display.waveshow(filtered_signal2, sr=sr2, color='r', ax=ax, alpha=0.7, label='音声ファイル2')
            ax.set_title(f"ノッチフィルタ適用信号（{remove_freq}Hzを除去）")
            ax.legend(['音声ファイル1', '音声ファイル2'])
            self.analysis_data['ノッチフィルタ_音声ファイル2'] = filtered_signal2.tolist()
        else:
            ax.legend(['音声ファイル1'])

        canvas = FigureCanvas(fig)
        self.content_layout.addWidget(canvas)
        self.figures.append(fig)  # 保存用

    def handle_compare_analysis(self, y1, sr1, y2, sr2):
        # 比較分析のプロット
        if y1 is None or y2 is None:
            QMessageBox.warning(self, "警告", "比較分析には2つの音声ファイルが必要です。")
            return

        fig, ax = plt.subplots(figsize=(12, 6))
        librosa.display.waveshow(y1, sr=sr1, alpha=0.5, label='音声ファイル1')
        librosa.display.waveshow(y2, sr=sr2, color='r', alpha=0.5, label='音声ファイル2')
        ax.set_title("音声ファイル1と音声ファイル2の波形比較")
        ax.legend()
        canvas = FigureCanvas(fig)
        self.content_layout.addWidget(canvas)
        self.figures.append(fig)  # 保存用

        # 分析データの保存
        self.analysis_data['比較分析_音声ファイル1'] = y1.tolist()
        self.analysis_data['比較分析_音声ファイル2'] = y2.tolist()

    def handle_evaluate_similarity(self, y1, sr1, y2, sr2):
        # 類似度評価のプロット
        if y1 is None or y2 is None:
            QMessageBox.warning(self, "警告", "類似度評価には2つの音声ファイルが必要です。")
            return

        try:
            # MFCCを使用して類似度を計算
            mfcc1 = librosa.feature.mfcc(y=y1, sr=sr1, n_mfcc=20)
            mfcc2 = librosa.feature.mfcc(y=y2, sr=sr2, n_mfcc=20)
            # フレーム数を合わせる
            min_frames = min(mfcc1.shape[1], mfcc2.shape[1])
            mfcc1 = mfcc1[:, :min_frames]
            mfcc2 = mfcc2[:, :min_frames]
            # コサイン類似度を計算
            similarity = cosine_similarity(mfcc1.T, mfcc2.T)
            avg_similarity = np.mean(similarity)

            # 類似度マトリックスをヒートマップで表示
            fig, ax = plt.subplots(figsize=(8, 6))
            cax = ax.imshow(similarity, aspect='auto', origin='lower', cmap='viridis')
            fig.colorbar(cax)
            ax.set_title('MFCCベースの類似度マトリックス')
            ax.set_xlabel('音声ファイル2のフレーム')
            ax.set_ylabel('音声ファイル1のフレーム')
            ax.text(0.5, 1.05, f"音声ファイル間の平均類似度（MFCCベース）：{avg_similarity:.4f}", 
                    transform=ax.transAxes, ha='center', va='bottom', fontsize=10)
            canvas = FigureCanvas(fig)
            self.content_layout.addWidget(canvas)
            self.figures.append(fig)  # 保存用

            # 分析データの保存
            self.analysis_data['類似度評価_類似度マトリックス'] = similarity.tolist()
            self.analysis_data['類似度評価_平均類似度'] = avg_similarity
        except Exception as e:
            self.error.emit(f"類似度評価中にエラーが発生しました: {str(e)}")

    def handle_detect_anomalies(self, y1, sr1, y2, sr2):
        # 異常検知のプロット
        def anomaly_detection(y, sr, file_name):
            # エネルギーベースの異常検知（閾値を設定）
            rmse = librosa.feature.rms(y=y)[0]
            frames = range(len(rmse))
            times = librosa.frames_to_time(frames, sr=sr)
            threshold = np.mean(rmse) + 2 * np.std(rmse)
            anomalies = rmse > threshold
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(times, rmse, label='エネルギー')
            ax.hlines(threshold, times[0], times[-1], colors='r', linestyles='dashed', label='閾値')
            ax.set_title(f"{file_name}のエネルギーと異常検知")
            ax.set_xlabel("時間 (秒)")
            ax.set_ylabel("エネルギー")
            ax.legend()
            canvas = FigureCanvas(fig)
            self.content_layout.addWidget(canvas)
            self.figures.append(fig)  # 保存用

            # 分析データの保存
            self.analysis_data[f'異常検知_{file_name}'] = {
                '時間': times.tolist(),
                'エネルギー': rmse.tolist(),
                '閾値': threshold
            }

        if y1 is not None:
            anomaly_detection(y1, sr1, "音声ファイル1")
        if y2 is not None:
            anomaly_detection(y2, sr2, "音声ファイル2")

    def handle_advanced_visualization(self, y1, sr1, y2, sr2):
        # 高度な可視化（3Dスペクトログラム）のプロット
        def plot_3d_spectrogram(y, sr, file_name):
            D = librosa.stft(y)
            S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
            fig = plt.figure(figsize=(12, 6))
            ax = fig.add_subplot(111, projection='3d')
            X, Y = np.meshgrid(np.arange(S_db.shape[1]), np.arange(S_db.shape[0]))
            ax.plot_surface(X, Y, S_db, cmap='viridis', linewidth=0, antialiased=False)
            ax.set_xlabel('時間フレーム')
            ax.set_ylabel('周波数ビン')
            ax.set_zlabel('振幅 (dB)')
            ax.set_title(f"{file_name}の3Dスペクトログラム")
            canvas = FigureCanvas(fig)
            self.content_layout.addWidget(canvas)
            self.figures.append(fig)  # 保存用

            # 分析データの保存
            self.analysis_data[f'高度な可視化_{file_name}'] = S_db.tolist()

        if y1 is not None:
            plot_3d_spectrogram(y1, sr1, "音声ファイル1")
        if y2 is not None:
            plot_3d_spectrogram(y2, sr2, "音声ファイル2")

    def handle_show_statistics(self, y1, sr1, y2, sr2):
        # 統計情報の表示
        def display_stats(y, sr, file_name):
            duration = librosa.get_duration(y=y, sr=sr)
            mean = np.mean(y)
            std = np.std(y)
            max_amp = np.max(y)
            min_amp = np.min(y)
            stats_text = f"**{file_name}の統計情報**\n" \
                        f"- 長さ（秒）：{duration:.2f}\n" \
                        f"- 平均振幅：{mean:.4f}\n" \
                        f"- 振幅の標準偏差：{std:.4f}\n" \
                        f"- 最大振幅：{max_amp:.4f}\n" \
                        f"- 最小振幅：{min_amp:.4f}\n"
            self.result_area.append(stats_text)

            # 分析データの保存
            self.analysis_data[f'統計情報_{file_name}'] = {
                '長さ（秒）': duration,
                '平均振幅': mean,
                '振幅の標準偏差': std,
                '最大振幅': max_amp,
                '最小振幅': min_amp
            }

        if y1 is not None:
            display_stats(y1, sr1, "音声ファイル1")
        if y2 is not None:
            display_stats(y2, sr2, "音声ファイル2")

    def save_images(self):
        if not self.figures:
            QMessageBox.warning(self, "警告", "保存する画像がありません。")
            return

        # 保存先ディレクトリの選択
        options = QFileDialog.Options()
        directory = QFileDialog.getExistingDirectory(self, "保存先ディレクトリを選択", "", options=options)
        if not directory:
            return

        try:
            for idx, fig in enumerate(self.figures, start=1):
                # 各Figureを保存
                filename = f"analysis_result_{idx}.png"
                filepath = os.path.join(directory, filename)
                fig.savefig(filepath)
            QMessageBox.information(self, "完了", "分析結果の画像を保存しました。")
        except Exception as e:
            QMessageBox.critical(self, "エラー", f"画像保存中にエラーが発生しました: {str(e)}")

    def closeEvent(self, event):
        # アプリケーション終了時にバックグラウンドスレッドを終了
        try:
            self.thread.quit()
            self.thread.wait()
        except AttributeError:
            pass
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = AudioFeatureApp()
    ex.show()
    sys.exit(app.exec_())
