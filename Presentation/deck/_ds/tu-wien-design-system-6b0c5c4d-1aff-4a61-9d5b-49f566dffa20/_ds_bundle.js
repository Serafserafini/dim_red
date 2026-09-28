/* @ds-bundle: {"format":4,"namespace":"TUWienDesignSystem_6b0c5c","components":[{"name":"Logo","sourcePath":"components/brand/Logo.jsx"},{"name":"UrlLockup","sourcePath":"components/brand/UrlLockup.jsx"},{"name":"AgendaList","sourcePath":"components/content/AgendaList.jsx"},{"name":"BulletList","sourcePath":"components/content/BulletList.jsx"},{"name":"Button","sourcePath":"components/content/Button.jsx"},{"name":"ContactBlock","sourcePath":"components/content/ContactBlock.jsx"},{"name":"DataTable","sourcePath":"components/content/DataTable.jsx"},{"name":"Quote","sourcePath":"components/content/Quote.jsx"},{"name":"CopyrightNote","sourcePath":"components/slide/CopyrightNote.jsx"},{"name":"Slide","sourcePath":"components/slide/Slide.jsx"},{"name":"SlideBody","sourcePath":"components/slide/SlideBody.jsx"},{"name":"SlideFooter","sourcePath":"components/slide/SlideFooter.jsx"},{"name":"SlideTitle","sourcePath":"components/slide/SlideTitle.jsx"}],"sourceHashes":{"components/brand/Logo.jsx":"d283f6786bab","components/brand/UrlLockup.jsx":"2237856768f4","components/content/AgendaList.jsx":"9420787c24e5","components/content/BulletList.jsx":"bed77ebd9210","components/content/Button.jsx":"e053d208fe7e","components/content/ContactBlock.jsx":"c04fec45cd48","components/content/DataTable.jsx":"5890e1ea9194","components/content/Quote.jsx":"4c20fab1b47e","components/slide/CopyrightNote.jsx":"299f8d735704","components/slide/Slide.jsx":"35fdef31b104","components/slide/SlideBody.jsx":"e685bf677b5f","components/slide/SlideFooter.jsx":"8c667b98385d","components/slide/SlideTitle.jsx":"4dd3b0e2a709"},"inlinedExternals":[],"unexposedExports":[]} */

(() => {

const __ds_ns = (window.TUWienDesignSystem_6b0c5c = window.TUWienDesignSystem_6b0c5c || {});

const __ds_scope = {};

(__ds_ns.__errors = __ds_ns.__errors || []);

// components/brand/Logo.jsx
try { (() => {
const SRC = {
  full: {
    light: 'assets/logo-tuw-full.svg',
    dark: 'assets/logo-tuw-full-white.svg',
    ratio: 204 / 77
  },
  mark: {
    light: 'assets/logo-tuw-mark.svg',
    dark: 'assets/logo-tuw-mark.svg',
    ratio: 1
  }
};

/** TU Wien logo. Never redraw or recolour the mark — only these two files. */
function Logo({
  variant = 'full',
  tone = 'light',
  height,
  width,
  base = '',
  style,
  ...rest
}) {
  const cfg = SRC[variant] || SRC.full;
  const h = height != null ? height : variant === 'mark' ? 51 : 77;
  const w = width != null ? width : Math.round(h * cfg.ratio);
  return React.createElement('img', {
    src: (base ? base.replace(/\/$/, '') + '/' : '') + (tone === 'dark' ? cfg.dark : cfg.light),
    alt: 'TU Wien',
    width: w,
    height: h,
    style: {
      display: 'block',
      width: w + 'px',
      height: h + 'px',
      ...style
    },
    ...rest
  });
}
Object.assign(__ds_scope, { Logo });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/brand/Logo.jsx", error: String((e && e.message) || e) }); }

// components/brand/UrlLockup.jsx
try { (() => {
/** The white "www.tuwien.at" line that sits in the footer band of title slides. */
function UrlLockup({
  href = 'https://www.tuwien.at',
  children = 'www.tuwien.at',
  align = 'right',
  style,
  ...rest
}) {
  return React.createElement('a', {
    href,
    style: {
      display: 'block',
      textAlign: align,
      textDecoration: 'none',
      font: 'var(--weight-regular) var(--slide-url)/1.2 var(--font-sans)',
      color: 'var(--text-inverse)',
      letterSpacing: 'var(--tracking-normal)',
      ...style
    },
    ...rest
  }, children);
}
Object.assign(__ds_scope, { UrlLockup });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/brand/UrlLockup.jsx", error: String((e && e.message) || e) }); }

// components/content/AgendaList.jsx
try { (() => {
/** Agenda / table-of-contents list. Level-1 entries in bold TU blue, level-2 in black. */
function AgendaList({
  items = [],
  style,
  ...rest
}) {
  return React.createElement('ol', {
    style: {
      listStyle: 'none',
      margin: 0,
      padding: 0,
      ...style
    },
    ...rest
  }, items.map((it, i) => {
    const sub = it.level === 2;
    return React.createElement('li', {
      key: i,
      style: {
        display: 'flex',
        gap: '12px',
        alignItems: 'baseline',
        marginLeft: sub ? '28px' : 0,
        marginTop: (i === 0 ? 0 : sub ? 6 : 14) + 'px',
        fontSize: sub ? 'var(--slide-body-2)' : 'var(--slide-body-1)',
        lineHeight: 'var(--leading-tight)',
        fontWeight: sub ? 'var(--weight-regular)' : 'var(--weight-bold)',
        color: sub ? 'var(--text-body)' : 'var(--text-heading)',
        opacity: it.dimmed ? .45 : 1
      }
    }, it.number != null ? React.createElement('span', {
      style: {
        flex: '0 0 auto',
        minWidth: '1.4em',
        fontVariantNumeric: 'tabular-nums'
      }
    }, it.number) : React.createElement('span', {
      'aria-hidden': 'true',
      style: {
        flex: '0 0 auto',
        width: '.55em',
        height: '.55em',
        background: 'var(--bullet-color)',
        transform: 'translateY(-.05em)'
      }
    }), React.createElement('span', null, it.text));
  }));
}
Object.assign(__ds_scope, { AgendaList });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/content/AgendaList.jsx", error: String((e && e.message) || e) }); }

// components/content/BulletList.jsx
try { (() => {
const SIZE = {
  1: 'var(--slide-body-1)',
  2: 'var(--slide-body-2)',
  3: 'var(--slide-body-3)',
  4: 'var(--slide-body-3)',
  5: 'var(--slide-body-3)'
};
const INDENT = {
  1: 0,
  2: 28,
  3: 56,
  4: 84,
  5: 112
};

/** Bulleted body copy. The bullet is a solid TU-blue square (■), never a dot. */
function BulletList({
  items = [],
  tone,
  style,
  ...rest
}) {
  return React.createElement('ul', {
    style: {
      listStyle: 'none',
      margin: 0,
      padding: 0,
      ...style
    },
    ...rest
  }, items.map((it, i) => {
    const lvl = it.level || 1;
    const size = SIZE[lvl] || SIZE[3];
    return React.createElement('li', {
      key: i,
      style: {
        display: 'flex',
        gap: 'var(--gap-bullet)',
        alignItems: 'baseline',
        marginLeft: INDENT[lvl] + 'px',
        marginTop: (i === 0 ? 0 : lvl === 1 ? 8 : 5) + 'px',
        fontSize: size,
        lineHeight: 'var(--leading-tight)',
        fontWeight: it.bold ? 'var(--weight-bold)' : 'var(--weight-regular)',
        color: tone === 'inverse' ? 'var(--text-inverse)' : 'var(--text-body)'
      }
    }, it.bullet === false ? null : React.createElement('span', {
      'aria-hidden': 'true',
      style: {
        flex: '0 0 auto',
        width: '.55em',
        height: '.55em',
        background: tone === 'inverse' ? 'var(--text-inverse)' : 'var(--bullet-color)',
        transform: 'translateY(-.05em)'
      }
    }), React.createElement('span', {
      style: {
        textWrap: 'pretty'
      }
    }, it.text));
  }));
}
Object.assign(__ds_scope, { BulletList });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/content/BulletList.jsx", error: String((e && e.message) || e) }); }

// components/content/Button.jsx
try { (() => {
const TONES = {
  primary: {
    background: 'var(--surface-accent)',
    color: 'var(--text-inverse)',
    border: '1px solid var(--surface-accent)'
  },
  secondary: {
    background: 'transparent',
    color: 'var(--tuw-blue)',
    border: '1px solid var(--tuw-blue)'
  },
  ghost: {
    background: 'transparent',
    color: 'var(--tuw-blue)',
    border: '1px solid transparent'
  }
};
const SIZES = {
  md: {
    padding: '10px 20px',
    fontSize: 'var(--text-sm)'
  },
  sm: {
    padding: '6px 14px',
    fontSize: 'var(--text-xs)'
  },
  lg: {
    padding: '14px 28px',
    fontSize: 'var(--text-md)'
  }
};

/** Square-cornered action control in TU blue. See readme "Intentional additions". */
function Button({
  variant = 'primary',
  size = 'md',
  disabled = false,
  as = 'button',
  children,
  style,
  ...rest
}) {
  return React.createElement(as, {
    disabled: as === 'button' ? disabled : undefined,
    style: {
      display: 'inline-flex',
      alignItems: 'center',
      justifyContent: 'center',
      gap: '8px',
      fontFamily: 'var(--font-sans)',
      fontWeight: 'var(--weight-bold)',
      lineHeight: 1.2,
      borderRadius: 'var(--radius-sharp)',
      cursor: disabled ? 'default' : 'pointer',
      textDecoration: 'none',
      opacity: disabled ? .4 : 1,
      transition: 'background var(--motion-fast) var(--motion-ease),color var(--motion-fast) var(--motion-ease)',
      ...TONES[variant],
      ...SIZES[size],
      ...style
    },
    ...rest
  }, children);
}
Object.assign(__ds_scope, { Button });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/content/Button.jsx", error: String((e && e.message) || e) }); }

// components/content/ContactBlock.jsx
try { (() => {
/** The closing "Kontakt" block: name in bold, address lines, then phone and mail. */
function ContactBlock({
  name,
  unit,
  lines = [],
  phone,
  email,
  website,
  style,
  ...rest
}) {
  const line = {
    margin: 0,
    fontSize: 'var(--slide-body-2)',
    lineHeight: 1.35,
    color: 'var(--text-body)'
  };
  return React.createElement('address', {
    style: {
      fontStyle: 'normal',
      fontFamily: 'var(--font-sans)',
      ...style
    },
    ...rest
  }, name ? React.createElement('p', {
    style: {
      ...line,
      fontWeight: 'var(--weight-bold)'
    }
  }, name) : null, unit ? React.createElement('p', {
    style: line
  }, unit) : null, lines.map((l, i) => React.createElement('p', {
    key: i,
    style: line
  }, l)), phone || email || website ? React.createElement('div', {
    style: {
      marginTop: '16px'
    }
  }, phone ? React.createElement('p', {
    style: line
  }, 'Telefon: ', React.createElement('a', {
    href: 'tel:' + String(phone).replace(/[^+\d]/g, ''),
    style: {
      color: 'var(--text-link)'
    }
  }, phone)) : null, email ? React.createElement('p', {
    style: line
  }, React.createElement('a', {
    href: 'mailto:' + email,
    style: {
      color: 'var(--text-link)'
    }
  }, email)) : null, website ? React.createElement('p', {
    style: line
  }, React.createElement('a', {
    href: 'https://' + String(website).replace(/^https?:\/\//, ''),
    style: {
      color: 'var(--text-link)'
    }
  }, website)) : null) : null);
}
Object.assign(__ds_scope, { ContactBlock });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/content/ContactBlock.jsx", error: String((e && e.message) || e) }); }

// components/content/DataTable.jsx
try { (() => {
/**
 * TU Wien data table. Variant 1 has a filled TU-blue header row; variant 2 drops
 * the fill and uses a blue rule instead. Rows band with accent1 at 40%.
 */
function DataTable({
  columns = [],
  rows = [],
  totalRow,
  variant = 1,
  caption,
  style,
  ...rest
}) {
  const cell = {
    padding: '8px 12px',
    fontSize: 'var(--slide-table)',
    lineHeight: 1.25,
    fontFamily: 'var(--font-sans)'
  };
  const num = i => i > 0 ? {
    textAlign: 'right',
    fontVariantNumeric: 'tabular-nums'
  } : {
    textAlign: 'left'
  };
  return React.createElement('table', {
    style: {
      borderCollapse: 'collapse',
      width: '100%',
      color: 'var(--text-body)',
      ...style
    },
    ...rest
  }, caption ? React.createElement('caption', {
    style: {
      captionSide: 'top',
      textAlign: 'left',
      ...cell,
      paddingLeft: 0,
      color: 'var(--text-muted)'
    }
  }, caption) : null, React.createElement('thead', null, React.createElement('tr', null, columns.map((c, i) => React.createElement('th', {
    key: i,
    scope: 'col',
    style: {
      ...cell,
      ...num(i),
      fontWeight: 'var(--weight-bold)',
      background: variant === 1 ? 'var(--surface-accent)' : 'transparent',
      color: variant === 1 ? 'var(--text-inverse)' : 'var(--text-heading)',
      borderBottom: variant === 1 ? '2px solid var(--tuw-white)' : '2px solid var(--border-default)'
    }
  }, c)))), React.createElement('tbody', null, rows.map((r, ri) => React.createElement('tr', {
    key: ri,
    style: {
      background: ri % 2 === 0 ? 'var(--tuw-blue-band)' : 'transparent'
    }
  }, r.map((v, ci) => React.createElement(ci === 0 ? 'th' : 'td', {
    key: ci,
    scope: ci === 0 ? 'row' : undefined,
    style: {
      ...cell,
      ...num(ci),
      fontWeight: ci === 0 ? 'var(--weight-bold)' : 'var(--weight-regular)',
      borderRight: ci === 0 ? '2px solid var(--border-default)' : '1px solid var(--border-default)',
      borderBottom: '1px solid var(--border-default)'
    }
  }, v))))), totalRow ? React.createElement('tfoot', null, React.createElement('tr', null, totalRow.map((v, ci) => React.createElement('td', {
    key: ci,
    style: {
      ...cell,
      ...num(ci),
      fontWeight: 'var(--weight-bold)',
      borderTop: '2px solid var(--border-default)',
      borderBottom: '2px solid var(--border-default)'
    }
  }, v)))) : null);
}
Object.assign(__ds_scope, { DataTable });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/content/DataTable.jsx", error: String((e && e.message) || e) }); }

// components/content/Quote.jsx
try { (() => {
/** Full-bleed quote block: centred white 40pt bold on TU blue, attribution below right. */
function Quote({
  children,
  attribution,
  style,
  ...rest
}) {
  return React.createElement('div', {
    style: {
      position: 'absolute',
      left: '137px',
      top: '127px',
      width: '1006px',
      ...style
    },
    ...rest
  }, React.createElement('blockquote', {
    style: {
      margin: 0,
      height: '325px',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      textAlign: 'center',
      fontFamily: 'var(--font-sans)',
      fontWeight: 'var(--weight-bold)',
      fontSize: 'var(--slide-quote)',
      lineHeight: 'var(--leading-tight)',
      color: 'var(--text-inverse)',
      textWrap: 'balance'
    }
  }, children), attribution ? React.createElement('p', {
    style: {
      margin: '21px 0 0',
      width: '489px',
      marginLeft: '517px',
      font: 'var(--weight-regular) var(--slide-body-3)/1.3 var(--font-sans)',
      color: 'var(--text-inverse)'
    }
  }, attribution) : null);
}
Object.assign(__ds_scope, { Quote });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/content/Quote.jsx", error: String((e && e.message) || e) }); }

// components/slide/CopyrightNote.jsx
try { (() => {
/** 8pt source/copyright line sitting just above the footer band. */
function CopyrightNote({
  children = '© Copyright- oder Quellenhinweis, …',
  tone,
  style,
  ...rest
}) {
  return React.createElement('p', {
    style: {
      position: 'absolute',
      left: 'var(--slide-content-x)',
      top: 'var(--slide-copyright-y)',
      width: 'var(--slide-content-w)',
      margin: 0,
      font: 'var(--weight-regular) var(--slide-caption)/1.3 var(--font-sans)',
      color: tone === 'inverse' ? 'var(--text-inverse)' : 'var(--tuw-blue)',
      ...style
    },
    ...rest
  }, children);
}
Object.assign(__ds_scope, { CopyrightNote });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/slide/CopyrightNote.jsx", error: String((e && e.message) || e) }); }

// components/slide/SlideBody.jsx
try { (() => {
/** Positioned content well matching the template's body placeholder rectangles. */
function SlideBody({
  variant = 'default',
  columns = 1,
  gap = 32,
  x,
  y,
  width,
  height,
  children,
  style,
  ...rest
}) {
  const compact = variant === 'compact';
  return React.createElement('div', {
    style: {
      position: 'absolute',
      left: x != null ? x + 'px' : 'var(--slide-content-x)',
      top: y != null ? y + 'px' : compact ? 'var(--slide-body-y-small)' : 'var(--slide-body-y)',
      width: width != null ? width + 'px' : 'var(--slide-content-w)',
      height: height != null ? height + 'px' : compact ? '438px' : 'var(--slide-body-h)',
      display: columns > 1 ? 'grid' : 'block',
      gridTemplateColumns: columns > 1 ? `repeat(${columns},minmax(0,1fr))` : undefined,
      gap: columns > 1 ? gap + 'px' : undefined,
      fontSize: 'var(--slide-body-1)',
      lineHeight: 'var(--leading-tight)',
      color: 'inherit',
      ...style
    },
    ...rest
  }, children);
}
Object.assign(__ds_scope, { SlideBody });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/slide/SlideBody.jsx", error: String((e && e.message) || e) }); }

// components/slide/SlideFooter.jsx
try { (() => {
/** The blue band across the bottom of every content slide, with running footer + page number. */
function SlideFooter({
  text,
  pageNumber,
  bare = false,
  style,
  ...rest
}) {
  const type = {
    font: 'var(--weight-regular) var(--slide-footer)/1.2 var(--font-sans)',
    color: 'var(--text-inverse)'
  };
  return React.createElement('div', {
    style: {
      position: 'absolute',
      left: 0,
      right: 0,
      top: 'var(--slide-bar-y)',
      height: 'var(--slide-bar-h)',
      background: bare ? 'transparent' : 'var(--surface-bar)',
      ...style
    },
    ...rest
  }, text ? React.createElement('div', {
    style: {
      position: 'absolute',
      left: 'var(--slide-margin)',
      top: '15px',
      ...type
    }
  }, text) : null, pageNumber != null ? React.createElement('div', {
    style: {
      position: 'absolute',
      right: 'var(--slide-margin)',
      top: '15px',
      textAlign: 'right',
      ...type
    }
  }, pageNumber) : null);
}
Object.assign(__ds_scope, { SlideFooter });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/slide/SlideFooter.jsx", error: String((e && e.message) || e) }); }

// components/slide/Slide.jsx
try { (() => {
/**
 * The 1280x720 TU Wien slide frame. Handles background, logo placement and the
 * blue footer band; children are positioned against the slide grid.
 */
function Slide({
  variant = 'default',
  logo,
  footer,
  pageNumber,
  image,
  children,
  base = '',
  style,
  ...rest
}) {
  const dark = variant === 'accent' || variant === 'photo';
  const showLogo = logo !== undefined ? logo : variant === 'blank' ? 'none' : variant === 'compact' ? 'mark' : 'full';
  const frame = {
    position: 'relative',
    width: 'var(--slide-w)',
    height: 'var(--slide-h)',
    flex: '0 0 auto',
    overflow: 'hidden',
    boxSizing: 'border-box',
    background: variant === 'accent' ? 'var(--surface-accent)' : 'var(--surface-page)',
    color: dark ? 'var(--text-inverse)' : 'var(--text-body)',
    fontFamily: 'var(--font-sans)',
    borderRadius: 'var(--radius-sharp)',
    ...style
  };
  return React.createElement('div', {
    'data-slide-variant': variant,
    style: frame,
    ...rest
  }, variant === 'photo' && image ? React.createElement('img', {
    src: image,
    alt: '',
    style: {
      position: 'absolute',
      inset: '0 0 auto 0',
      width: '100%',
      height: '637px',
      objectFit: 'cover'
    }
  }) : null, showLogo !== 'none' ? React.createElement('div', {
    style: {
      position: 'absolute',
      left: 'var(--slide-margin)',
      top: 'var(--slide-margin)'
    }
  }, React.createElement(__ds_scope.Logo, {
    base,
    variant: showLogo === 'mark' ? 'mark' : 'full',
    tone: dark ? 'dark' : 'light',
    height: showLogo === 'mark' ? 51 : 77
  })) : null, children, footer !== undefined || pageNumber !== undefined ? React.createElement(__ds_scope.SlideFooter, {
    text: footer,
    pageNumber,
    bare: variant === 'accent'
  }) : null);
}
Object.assign(__ds_scope, { Slide });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/slide/Slide.jsx", error: String((e && e.message) || e) }); }

// components/slide/SlideTitle.jsx
try { (() => {
const SIZES = {
  default: {
    top: 'var(--slide-title-y)',
    size: 'var(--slide-title)'
  },
  compact: {
    top: 'var(--slide-title-y-small)',
    size: 'var(--slide-title-small)'
  },
  hero: {
    top: '239px',
    size: 'var(--slide-title-hero)'
  },
  quote: {
    top: '127px',
    size: 'var(--slide-quote)'
  }
};

/** Slide headline: Arial bold, TU blue, 90% line spacing, left-aligned on the 137px column. */
function SlideTitle({
  variant = 'default',
  tone,
  children,
  style,
  ...rest
}) {
  const s = SIZES[variant] || SIZES.default;
  const quote = variant === 'quote';
  return React.createElement('h2', {
    style: {
      position: 'absolute',
      left: quote ? '137px' : 'var(--slide-content-x)',
      top: s.top,
      width: quote ? '1006px' : 'var(--slide-content-w)',
      height: quote ? '325px' : 'auto',
      margin: 0,
      display: 'flex',
      alignItems: quote ? 'center' : 'flex-start',
      justifyContent: quote ? 'center' : 'flex-start',
      textAlign: quote ? 'center' : 'left',
      fontFamily: 'var(--font-sans)',
      fontWeight: 'var(--weight-bold)',
      fontSize: s.size,
      lineHeight: 'var(--leading-tight)',
      color: tone === 'inverse' ? 'var(--text-inverse)' : 'var(--text-heading)',
      textWrap: 'pretty',
      ...style
    },
    ...rest
  }, children);
}
Object.assign(__ds_scope, { SlideTitle });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/slide/SlideTitle.jsx", error: String((e && e.message) || e) }); }

__ds_ns.Logo = __ds_scope.Logo;

__ds_ns.UrlLockup = __ds_scope.UrlLockup;

__ds_ns.AgendaList = __ds_scope.AgendaList;

__ds_ns.BulletList = __ds_scope.BulletList;

__ds_ns.Button = __ds_scope.Button;

__ds_ns.ContactBlock = __ds_scope.ContactBlock;

__ds_ns.DataTable = __ds_scope.DataTable;

__ds_ns.Quote = __ds_scope.Quote;

__ds_ns.CopyrightNote = __ds_scope.CopyrightNote;

__ds_ns.Slide = __ds_scope.Slide;

__ds_ns.SlideBody = __ds_scope.SlideBody;

__ds_ns.SlideFooter = __ds_scope.SlideFooter;

__ds_ns.SlideTitle = __ds_scope.SlideTitle;

})();
