"""
Theme configuration for Tkinter application, translated from theme.ts.
Provides color schemes, typography, spacing and other styling properties
compatible with Tkinter widgets.
"""

theme = {
    'colors': {
        'primary': {
            'main': '#D3694C',
            'light': '#e6b3a3',
            'dark': '#5a2d21',
            'text': '#1a0a06',
        },
        'secondary': {
            'main': '#009b74',
            'light': '#00b487',
            'dark': '#006b50',
            'text': '#ffffff',
        },
        'accent': {
            'blue': '#008ea4',
            'purple': '#7056c4',
            'pink': '#CE4985',
            'yellow': '#ad8200',
            'green': '#4d9900',
            'orange': '#ce6700',
        },
        'background': {
            'primary': '#EAE4E1',
            'secondary': '#e3dbd7',
            'paper': '#eee9e7',
            'contrast': '#f4f1f0',
        },
        'text': {
            'primary': '#36221d',
            'secondary': '#a69692',
            'disabled': '#a69692',  # Approximated from #36221d4d since Tkinter doesn't support alpha
        },
        'border': {
            'main': '#cbbbb4',
            'light': '#d5c9c3',
            'dark': '#c0aea5',
        },
        'status': {
            'success': '#51A200',
            'error': '#FF3A3A',
            'warning': '#ce6700',
            'info': '#009DB5',
        },
    },
    'typography': {
        'font_family': 'Segoe UI',  # Windows-compatible system font
        'font_size': {
            'xs': 8,     # 0.75rem ≈ 12px ≈ 8pt
            'sm': 9,     # 0.875rem ≈ 14px ≈ 9pt
            'base': 10,  # 1rem ≈ 16px ≈ 10pt
            'lg': 11,    # 1.125rem ≈ 18px ≈ 11pt
            'xl': 12,    # 1.25rem ≈ 20px ≈ 12pt
            '2xl': 14,   # 1.5rem ≈ 24px ≈ 14pt
            '3xl': 18,   # 1.875rem ≈ 30px ≈ 18pt
            '4xl': 24,   # 2.25rem ≈ 36px ≈ 24pt
        },
        'font_weight': {
            'normal': 'normal',    # 400
            'medium': 'normal',    # 500 (Tkinter only supports normal/bold)
            'semibold': 'bold',    # 600
            'bold': 'bold',        # 700
        },
    },
    'spacing': {
        'xs': 4,     # 0.25rem = 4px
        'sm': 8,     # 0.5rem = 8px
        'md': 16,    # 1rem = 16px
        'lg': 24,    # 1.5rem = 24px
        'xl': 32,    # 2rem = 32px
        '2xl': 48,   # 3rem = 48px
        '3xl': 64,   # 4rem = 64px
    },
}

def get_font(size_key='base', weight_key='normal'):
    """
    Helper function to create font tuples for Tkinter widgets.
    
    Args:
        size_key (str): Key from typography.font_size (e.g., 'base', 'lg')
        weight_key (str): Key from typography.font_weight (e.g., 'normal', 'bold')
    
    Returns:
        tuple: Font tuple compatible with Tkinter (family, size, weight)
    """
    return (
        theme['typography']['font_family'],
        theme['typography']['font_size'][size_key],
        theme['typography']['font_weight'][weight_key]
    )

def configure_styles(style):
    """
    Configure ttk styles using the theme.
    
    Args:
        style: ttk.Style instance to configure
    """
    # Frame style
    style.configure('TFrame',
        background=theme['colors']['background']['primary']
    )
    
    # Button style
    style.configure('TButton',
        background=theme['colors']['primary']['main'],
        foreground=theme['colors']['primary']['text']
    )
    style.map('TButton',
        background=[('active', theme['colors']['primary']['light'])]
    )
    
    # Entry style
    style.configure('TEntry',
        fieldbackground=theme['colors']['background']['paper'],
        foreground=theme['colors']['text']['primary']
    )
    
    # Label style
    style.configure('TLabel',
        background=theme['colors']['background']['primary'],
        foreground=theme['colors']['text']['primary']
    )
    
    # Custom styles
    style.configure('Primary.TButton',
        background=theme['colors']['primary']['main'],
        foreground=theme['colors']['primary']['text']
    )
    style.map('Primary.TButton',
        background=[('active', theme['colors']['primary']['light'])]
    )
    
    style.configure('Secondary.TButton',
        background=theme['colors']['secondary']['main'],
        foreground=theme['colors']['secondary']['text']
    )
    style.map('Secondary.TButton',
        background=[('active', theme['colors']['secondary']['light'])]
    )
    
    style.configure('Success.TButton',
        background=theme['colors']['status']['success'],
        foreground='white'
    )
    
    style.configure('Error.TButton',
        background=theme['colors']['status']['error'],
        foreground='white'
    ) 