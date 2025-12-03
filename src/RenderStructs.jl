struct SubStruct{V, H, MicroNWarps, MicroMWarps, NBlocks, MBlocks}

end

function SubStruct(V, H, MicroNWarps, MicroMWarps, NBlocks, MBlocks)
    return SubStruct{V, H, MicroNWarps, MicroMWarps, NBlocks, MBlocks}()
end