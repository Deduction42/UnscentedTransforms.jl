
function Base.show(io::IO, mime::MIME"text/plain", obj::UvGaussian)
    show(io, mime, mean(obj))
    print(io, " ± ")
    return show(io, mime, std(obj))
end

function Base.show(io::IO, mime::MIME"text/plain", obj::MvGaussian)
    show(io, mime, mean(obj))
    print(io, "\n± ")
    return show(io, mime, _lowermat(std(obj)))
end

_lowermat(m::Cholesky) = m.L 
_lowermat(m::Diagonal) = m
