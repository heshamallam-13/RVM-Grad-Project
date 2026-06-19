from pathlib import Path

p = Path("pi_touch_gui.py")
s = p.read_text()

# Slightly stronger fonts/colors
s = s.replace('TITLE_FONT = ("Helvetica", 20, "bold")', 'TITLE_FONT = ("Helvetica", 22, "bold")')
s = s.replace('BUTTON_FONT = ("Helvetica", 18, "bold")', 'BUTTON_FONT = ("Helvetica", 19, "bold")')

# Replace main padding
s = s.replace(
'''        main = tk.Frame(self.root, bg=BG_COLOR)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
''',
'''        main = tk.Frame(self.root, bg=BG_COLOR)
        main.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)
'''
)

# Replace title label with richer title block
s = s.replace(
'''        tk.Label(
            title_frame, text="♻️  EcoVend RVM", font=TITLE_FONT,
            bg=BG_COLOR, fg=ACCENT_GREEN
        ).pack(side=tk.LEFT)
''',
'''        tk.Label(
            title_frame,
            text="♻️  EcoVend RVM",
            font=TITLE_FONT,
            bg=BG_COLOR,
            fg=ACCENT_GREEN
        ).pack(side=tk.LEFT)

        tk.Label(
            title_frame,
            text="Smart Recycling • AI Detection • Green Rewards",
            font=("Helvetica", 10, "bold"),
            bg=BG_COLOR,
            fg="#8b949e"
        ).pack(side=tk.LEFT, padx=14)
'''
)

# Replace fps label styling
s = s.replace(
'''        self.fps_label = tk.Label(
            title_frame, text="FPS: --", font=LABEL_FONT,
            bg=BG_COLOR, fg=ACCENT_BLUE
        )
''',
'''        self.fps_label = tk.Label(
            title_frame,
            text="● LIVE  FPS: --",
            font=("Helvetica", 12, "bold"),
            bg=BG_COLOR,
            fg=ACCENT_GREEN
        )
'''
)

# Replace video label block with neon camera card
old_video = '''        # Video canvas (left)
        self.video_label = tk.Label(
            center, bg="#000000", text="Press START",
            fg=TEXT_COLOR, font=LABEL_FONT,
            relief=tk.FLAT, borderwidth=0
        )
        self.video_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
'''

new_video = '''        # Video canvas (left) inside neon card
        video_card = tk.Frame(
            center,
            bg=ACCENT_GREEN,
            highlightbackground="#2ea043",
            highlightthickness=2
        )
        video_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

        video_header = tk.Frame(video_card, bg="#0f2417")
        video_header.pack(fill=tk.X)

        tk.Label(
            video_header,
            text="🎥 LIVE AI DETECTION",
            font=("Helvetica", 13, "bold"),
            bg="#0f2417",
            fg=ACCENT_GREEN
        ).pack(side=tk.LEFT, padx=10, pady=4)

        tk.Label(
            video_header,
            text="Place item on conveyor",
            font=("Helvetica", 10, "bold"),
            bg="#0f2417",
            fg="#8b949e"
        ).pack(side=tk.RIGHT, padx=10)

        self.video_label = tk.Label(
            video_card,
            bg="#000000",
            text="Press START",
            fg=TEXT_COLOR,
            font=("Helvetica", 18, "bold"),
            relief=tk.FLAT,
            borderwidth=0
        )
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 4))
'''
s = s.replace(old_video, new_video)

# Sidebar width and style
s = s.replace(
'''        sidebar = tk.Frame(center, bg=CARD_BG, width=220)
''',
'''        sidebar = tk.Frame(center, bg="#101820", width=235)
'''
)

# POINTS label
s = s.replace(
'''        tk.Label(
            sidebar, text="POINTS", font=LABEL_FONT,
            bg=CARD_BG, fg=ACCENT_YELLOW
        ).pack(pady=(15, 0))
''',
'''        tk.Label(
            sidebar,
            text="🏆 LIVE SCORE",
            font=("Helvetica", 15, "bold"),
            bg="#101820",
            fg=ACCENT_YELLOW
        ).pack(pady=(12, 0))
'''
)

# Canvas background
s = s.replace('bg=CARD_BG,\n            highlightthickness=0\n        )\n        self.points_canvas.pack', 'bg="#101820",\n            highlightthickness=0\n        )\n        self.points_canvas.pack')

# PET/CAN labels become card-like
s = s.replace(
'''        self.pet_label = tk.Label(
            sidebar, text="🥤 PET: 0", font=LABEL_FONT,
            bg=CARD_BG, fg=ACCENT_GREEN
        )
        self.pet_label.pack(pady=2)

        self.can_label = tk.Label(
            sidebar, text="🥫 CAN: 0", font=LABEL_FONT,
            bg=CARD_BG, fg=ACCENT_BLUE
        )
        self.can_label.pack(pady=2)
''',
'''        self.pet_label = tk.Label(
            sidebar,
            text="🥤 PET: 0",
            font=("Helvetica", 14, "bold"),
            bg="#0f2417",
            fg=ACCENT_GREEN,
            padx=8,
            pady=6
        )
        self.pet_label.pack(fill=tk.X, padx=10, pady=4)

        self.can_label = tk.Label(
            sidebar,
            text="🥫 CAN: 0",
            font=("Helvetica", 14, "bold"),
            bg="#0d1b2a",
            fg=ACCENT_BLUE,
            padx=8,
            pady=6
        )
        self.can_label.pack(fill=tk.X, padx=10, pady=4)
'''
)

# Separator color bg from CARD_BG not critical; update det/status bg
s = s.replace('bg=CARD_BG, fg=TEXT_COLOR, wraplength=200', 'bg="#101820", fg=TEXT_COLOR, wraplength=210')
s = s.replace('bg=CARD_BG, fg="#8b949e", wraplength=200', 'bg="#101820", fg="#8b949e", wraplength=210')

# Button frame and buttons
s = s.replace(
'''        btn_frame = tk.Frame(main, bg=BG_COLOR)
        btn_frame.pack(fill=tk.X, pady=(5, 5))
''',
'''        btn_frame = tk.Frame(main, bg=BG_COLOR)
        btn_frame.pack(fill=tk.X, pady=(8, 2))
'''
)

s = s.replace('text="▶  START"', 'text="▶  START RECYCLING"')
s = s.replace('text="⏭  NEXT"', 'text="⏭  ACCEPT ITEM"')
s = s.replace('text="⏹  FINISH"', 'text="⏹  FINISH SESSION"')

s = s.replace('padx=20, pady=12,\n            command=self.start_detection', 'padx=20, pady=14,\n            command=self.start_detection')
s = s.replace('padx=20, pady=12,\n            state=tk.DISABLED, command=self.next_item', 'padx=20, pady=14,\n            state=tk.DISABLED, command=self.next_item')
s = s.replace('padx=20, pady=12,\n            state=tk.DISABLED, command=self.finish_session', 'padx=20, pady=14,\n            state=tk.DISABLED, command=self.finish_session')

p.write_text(s)
print("Layout polish patch applied successfully.")
