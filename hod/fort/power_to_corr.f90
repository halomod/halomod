module hankel
    use utils
    implicit none

    real(8), parameter :: pi=3.141592653589

    contains

    real(8) function interp(in,x,y)
        real(8), intent(in) :: in
        real(8), intent(in) :: x(:), y(:)
        !real(8), intent(out) :: interp

        real(8) :: dx, slope
        integer :: bin

        !Check if in is within bounds
        if(in.lt.x(1))then
            write(*,*) "WARNING: INTERPOLATING BELOW BOUNDS!"
            interp = 0.d0

        elseif(in.ge.x(size(x)))then
            write(*,*) "WARNING: INTERPOLATING ABOVE BOUNDS!"
            interp = 0.d0

        else
            dx = x(2) - x(1)
            bin = int(ceiling((in - x(1))/dx))
            slope = (y(bin+1)-y(bin))/dx
            interp = y(bin) + (in-x(bin))*slope
        end if
    end function interp

    subroutine power_to_corr_singler(nk,power,r,lnk,corr)
        integer, intent(in) :: nk
        real(8), intent(in) :: r, power(nk), lnk(nk)
        real(8), intent(out) :: corr

        integer ::  j,minsteps
        real(8) :: min_k, temp_min_k, dlnk,lnkval,maxk,mink

        real(8), allocatable :: thislnk(:), thispower(:), integ(:)
        integer :: thisnk

        ! Set some parameters
        minsteps = 6 !number of steps to fit into half-period at high-k. 6 < 1e-4
        min_k = exp(lnk(1)) !Where to integrate from.
        temp_min_k = 1.0d0

        ! Rather than continual re-allocation, set max size for thislnk
        mink = (2 * ceiling((temp_min_k * r / pi - 1.d0) / 2.d0) + 1.d0) * pi / r
        maxk = max(39*pi/r,mink)
        thisnk = ceiling(log(maxk/min_k)/log(maxk/(maxk-pi/(minsteps*r))))
        allocate(thislnk(thisnk),thispower(thisnk),integ(thisnk))

        lnkval = log(min_k)
        dlnk = (log(maxk) - lnkval)/thisnk

        do j = 1,thisnk
            lnkval = lnkval+dlnk
            thislnk(j) = lnkval
            thispower(j) = interp(lnkval,lnk,power)
        end do
        integ = thispower * exp(thislnk)**2 * sin(exp(thislnk)*r)/r
        call simps(dlnk,integ,corr)
        corr = (0.5/pi**2) * corr
    end subroutine power_to_corr_singler

    subroutine power_to_corr(nr,nk,r,power,lnk,corr)
        integer, intent(in) :: nr, nk
        real(8), intent(in) :: r(nr), power(nk), lnk(nk)
        real(8), intent(out) :: corr(nr)

        integer :: i, j,minsteps
        real(8) :: min_k, temp_min_k, dlnk,lnkval,maxk,mink,rr

        real(8), allocatable :: thislnk(:), thispower(:), integ(:)
        integer :: thisnk

        ! Set some parameters
        minsteps = 6 !number of steps to fit into half-period at high-k. 6 < 1e-4
        min_k = exp(lnk(1)) !Where to integrate from.
        temp_min_k = 1.0d0

        ! Rather than continual re-allocation, set max size for thislnk
        rr = r(1)
        mink = (2 * ceiling((temp_min_k * rr / pi - 1.d0) / 2.d0) + 1.d0) * pi / rr
        maxk = max(39*pi/rr,mink)
        thisnk = ceiling(log(maxk/min_k)/log(maxk/(maxk-pi/(minsteps*rr))))
        !write(*,*) "thisnk max:", thisnk
        allocate(thislnk(thisnk),thispower(thisnk),integ(thisnk))
        corr = 0.d0
        do i=1,nr
            rr = r(i)
            !getting maxk here is the important part. It must be an odd multiple of
            !pi/r to be at a "zero", it must be >1 AND it must have a number of half
            ! cycles > 38 (for 1E-5 precision).

            !mink is the minimum value that mak can be
            mink = (2 * ceiling((temp_min_k * rr / pi - 1.d0) / 2.d0) + 1.d0) * pi / rr
            maxk = max(39*pi/rr,mink)

            thisnk = ceiling(log(maxk/min_k)/log(maxk/(maxk-pi/(minsteps*rr))))

            lnkval = log(min_k)
            dlnk = (log(maxk) - lnkval)/thisnk
            write(*,*) r(i), thisnk, min_k, maxk

            do j = 1,thisnk
                lnkval = lnkval+dlnk
                thislnk(j) = lnkval
                thispower(j) = interp(lnkval,lnk,power)
            end do
            integ = thispower * exp(thislnk)**2 * sin(exp(thislnk)*rr)/rr
            call simps(dlnk,integ(1:thisnk),corr(i))
            corr(i) = (0.5/pi**2) * corr(i)
        end do
    end subroutine power_to_corr

    subroutine power_to_corr_matrix(nr,nk,r,lnk,power,corr)
        ! Power to corr, but for a matrix of power (nk,nr).
        ! In this case the power is scale-dependendent
        integer, intent(in) :: nr, nk
        real(8), intent(in) :: r(nr), power(nk,nr), lnk(nk)
        real(8), intent(out) :: corr(nr)

        integer :: i, j,minsteps
        real(8) :: min_k, temp_min_k, dlnk,lnkval,maxk,mink,rr

        real(8), allocatable :: thislnk(:), thispower(:), integ(:)
        integer :: thisnk

        ! Set some parameters
        minsteps = 6 !number of steps to fit into half-period at high-k. 6 < 1e-4
        min_k = exp(lnk(1)) !Where to integrate from.
        temp_min_k = 1.0d0

        ! Rather than continual re-allocation, set max size for thislnk
        rr = r(1)
        mink = (2 * ceiling((temp_min_k * rr / pi - 1.d0) / 2.d0) + 1.d0) * pi / rr
        maxk = max(39*pi/rr,mink)
        thisnk = ceiling(log(maxk/min_k)/log(maxk/(maxk-pi/(minsteps*rr))))
        !write(*,*) "thisnk max:", thisnk
        allocate(thislnk(thisnk),thispower(thisnk),integ(thisnk))
        corr = 0.d0

        do i=1,nr
            rr = r(i)
            !getting maxk here is the important part. It must be an odd multiple of
            !pi/r to be at a "zero", it must be >1 AND it must have a number of half
            ! cycles > 38 (for 1E-5 precision).

            !mink is the minimum value that mak can be
            mink = (2 * ceiling((temp_min_k * rr / pi - 1.d0) / 2.d0) + 1.d0) * pi / rr
            maxk = max(39*pi/rr,mink)

            thisnk = ceiling(log(maxk/min_k)/log(maxk/(maxk-pi/(minsteps*rr))))

            lnkval = log(min_k)
            dlnk = (log(maxk) - lnkval)/thisnk
            write(*,*) r(i), thisnk, min_k, maxk

            do j = 1,thisnk
                lnkval = lnkval+dlnk
                thislnk(j) = lnkval
                thispower(j) = interp(lnkval,lnk,power(:,i))
            end do

            integ = thispower * exp(thislnk)**2 * sin(exp(thislnk)*rr)/rr
            call simps(dlnk,integ(1:thisnk),corr(i))
            corr(i) = (0.5/pi**2) * corr(i)
        end do
    end subroutine power_to_corr_matrix

end module hankel
