fredmd_targets <- data.frame(
  id = c(
    # Prices (tcode 6) -- 20 series
    "WPSFD49207", "WPSFD49502", "WPSID61", "WPSID62", "OILPRICEx", "PPICMM",
    "CPIAUCSL", "CPIAPPSL", "CPITRNSL", "CPIMEDSL", "CUSR0000SAC", "CUSR0000SAD",
    "CUSR0000SAS", "CPIULFSL", "CUSR0000SA0L2", "CUSR0000SA0L5", "PCEPI",
    "DDURRG3M086SBEA", "DNDGRG3M086SBEA", "DSERRG3M086SBEA",
    # Rate levels (tcode 2) -- 9 series
    "FEDFUNDS", "CP3Mx", "TB3MS", "TB6MS", "GS1", "GS5", "GS10", "AAA", "BAA",
    # Credit/term spreads (tcode 1) -- 8 series
    "COMPAPFFx", "TB3SMFFM", "TB6SMFFM", "T1YFFM", "T5YFFM", "T10YFFM", "AAAFFM", "BAAFFM",
    # FX rates (tcode 5) -- 4 series
    "EXSZUSx", "EXJPUSx", "EXUSUKx", "EXCAUSx"
  ),
  group = c(
    rep("Prices", 20),
    rep("Rate", 9),
    rep("Spread", 8),
    rep("FX", 4)
  ),
  tcode = c(
    rep(6, 20),
    rep(2, 9),
    rep(1, 8),
    rep(5, 4)
  ),
  stringsAsFactors = FALSE
)
